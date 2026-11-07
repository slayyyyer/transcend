package org.transcend.nmt;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Three ONNX graphs: encoder, decoder init (builds the cross-attention cache)
 * and decoder step (one new token, reuses both caches).
 */
final class OnnxNmtModel implements AutoCloseable {
    private final OrtEnvironment environment;
    private final OrtSession encoder;
    private final OrtSession decoderInit;
    private final OrtSession decoderStep;
    private final String[] cacheNames;
    private final int layers;

    record Init(float[][] logits, KvCache cache) {
    }

    record Step(float[][] logits, KvCache cache) {
    }

    OnnxNmtModel(Path modelDirectory, int intraOpThreads) throws OrtException, IOException {
        environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        if (intraOpThreads > 0) {
            options.setIntraOpNumThreads(intraOpThreads);
        }
        options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        encoder = environment.createSession(modelDirectory.resolve("onnx/encoder.onnx").toString(), options);
        decoderInit = environment.createSession(modelDirectory.resolve("onnx/decoder_init.onnx").toString(), options);
        decoderStep = environment.createSession(modelDirectory.resolve("onnx/decoder_step.onnx").toString(), options);
        List<String> names = new ArrayList<>();
        for (int index = 0; index < 64; index++) {
            if (!hasInput(decoderStep, "past_self_key_" + index)) {
                break;
            }
            names.add("self_key_" + index);
            names.add("self_value_" + index);
            names.add("cross_key_" + index);
            names.add("cross_value_" + index);
        }
        if (names.isEmpty()) {
            throw new IOException("decoder_step graph has no recognised cache inputs");
        }
        this.cacheNames = names.toArray(new String[0]);
        this.layers = names.size() / 4;
    }

    private static boolean hasInput(OrtSession session, String name) {
        return session.getInputNames().contains(name);
    }

    int layers() {
        return layers;
    }

    /** Encoder run for a single unpadded source sequence. */
    float[][][] encode(int[] sourceIds, int[] positionIds) throws OrtException {
        long[][] ids = {toLongArray(sourceIds)};
        boolean[][] padding = {new boolean[sourceIds.length]};
        long[][] positions = {toLongArray(positionIds)};
        Map<String, OnnxTensor> inputs = new HashMap<>();
        try (OnnxTensor idTensor = longs(ids);
             OnnxTensor padTensor = bools(padding);
             OnnxTensor posTensor = longs(positions)) {
            inputs.put("src_ids", idTensor);
            inputs.put("src_pad_mask", padTensor);
            inputs.put("position_ids", posTensor);
            try (OrtSession.Result result = encoder.run(inputs)) {
                return read3((OnnxTensor) result.get(0));
            }
        }
    }

    Init decodeInit(float[][][] memory) throws OrtException {
        int sourceLength = memory[0].length;
        long[][] tgt = {{ByteBpeTokenizer.BOS_ID}};
        long[][] positions = {{0L}};
        float[][][][] selfMask = new float[1][1][1][1];
        float[][][][] crossMask = new float[1][1][1][sourceLength];
        Map<String, OnnxTensor> inputs = new HashMap<>();
        try (OnnxTensor tgtTensor = longs(tgt);
             OnnxTensor memoryTensor = floats3(memory);
             OnnxTensor selfTensor = floats4(selfMask);
             OnnxTensor crossTensor = floats4(crossMask);
             OnnxTensor posTensor = longs(positions)) {
            inputs.put("tgt_ids", tgtTensor);
            inputs.put("memory", memoryTensor);
            inputs.put("self_mask", selfTensor);
            inputs.put("cross_mask", crossTensor);
            inputs.put("position_ids", posTensor);
            try (OrtSession.Result result = decoderInit.run(inputs)) {
                float[][] logits = squeeze(read3((OnnxTensor) result.get(0)));
                return new Init(logits, readCache(result, 1));
            }
        }
    }

    Step decodeStep(int[] tokens, int position, KvCache cache) throws OrtException {
        int batch = tokens.length;
        long[][] tgt = new long[batch][1];
        long[][] positions = new long[batch][1];
        for (int row = 0; row < batch; row++) {
            tgt[row][0] = tokens[row];
            positions[row][0] = position;
        }
        Map<String, OnnxTensor> inputs = new HashMap<>();
        List<OnnxTensor> tensors = new ArrayList<>();
        try {
            OnnxTensor tgtTensor = longs(tgt);
            OnnxTensor posTensor = longs(positions);
            tensors.add(tgtTensor);
            tensors.add(posTensor);
            inputs.put("tgt_ids", tgtTensor);
            inputs.put("position_ids", posTensor);
            for (int index = 0; index < layers; index++) {
                addCache(inputs, tensors, "past_self_key_" + index, cache.selfKey.get(index));
                addCache(inputs, tensors, "past_self_value_" + index, cache.selfValue.get(index));
                addCache(inputs, tensors, "past_cross_key_" + index, cache.crossKey.get(index));
                addCache(inputs, tensors, "past_cross_value_" + index, cache.crossValue.get(index));
            }
            try (OrtSession.Result result = decoderStep.run(inputs)) {
                float[][] logits = squeeze(read3((OnnxTensor) result.get(0)));
                return new Step(logits, readCache(result, batch));
            }
        } finally {
            for (OnnxTensor tensor : tensors) {
                tensor.close();
            }
        }
    }

    private void addCache(Map<String, OnnxTensor> inputs, List<OnnxTensor> tensors,
                          String name, float[][][][] value) throws OrtException {
        OnnxTensor tensor = floats4(value);
        tensors.add(tensor);
        inputs.put(name, tensor);
    }

    private KvCache readCache(OrtSession.Result result, int batch) throws OrtException {
        List<float[][][][]> selfKey = new ArrayList<>();
        List<float[][][][]> selfValue = new ArrayList<>();
        List<float[][][][]> crossKey = new ArrayList<>();
        List<float[][][][]> crossValue = new ArrayList<>();
        for (int index = 0; index < layers; index++) {
            selfKey.add(read4((OnnxTensor) result.get(1 + 4 * index)));
            selfValue.add(read4((OnnxTensor) result.get(2 + 4 * index)));
            crossKey.add(read4((OnnxTensor) result.get(3 + 4 * index)));
            crossValue.add(read4((OnnxTensor) result.get(4 + 4 * index)));
        }
        return new KvCache(selfKey, selfValue, crossKey, crossValue);
    }

    private static long[] toLongArray(int[] values) {
        long[] out = new long[values.length];
        for (int index = 0; index < values.length; index++) {
            out[index] = values[index];
        }
        return out;
    }

    private static float[][] squeeze(float[][][] input) {
        return input[0];
    }

    // ------------------------------------------------------------ tensor glue
    private OnnxTensor longs(long[][] data) throws OrtException {
        int rows = data.length;
        int cols = data[0].length;
        LongBuffer buffer = LongBuffer.allocate(rows * cols);
        for (long[] row : data) {
            buffer.put(row);
        }
        buffer.rewind();
        return OnnxTensor.createTensor(environment, buffer, new long[]{rows, cols});
    }

    private OnnxTensor bools(boolean[][] data) throws OrtException {
        int rows = data.length;
        int cols = data[0].length;
        ByteBuffer buffer = ByteBuffer.allocate(rows * cols);
        for (boolean[] row : data) {
            for (boolean value : row) {
                buffer.put((byte) (value ? 1 : 0));
            }
        }
        buffer.rewind();
        return OnnxTensor.createTensor(environment, buffer, new long[]{rows, cols}, OnnxJavaType.BOOL);
    }

    private OnnxTensor floats3(float[][][] data) throws OrtException {
        int a = data.length;
        int b = data[0].length;
        int c = data[0][0].length;
        FloatBuffer buffer = FloatBuffer.allocate(a * b * c);
        for (float[][] plane : data) {
            for (float[] row : plane) {
                buffer.put(row);
            }
        }
        buffer.rewind();
        return OnnxTensor.createTensor(environment, buffer, new long[]{a, b, c});
    }

    private OnnxTensor floats4(float[][][][] data) throws OrtException {
        int a = data.length;
        int b = data[0].length;
        int c = data[0][0].length;
        int d = data[0][0][0].length;
        FloatBuffer buffer = FloatBuffer.allocate(a * b * c * d);
        for (float[][][] tensor : data) {
            for (float[][] head : tensor) {
                for (float[] row : head) {
                    buffer.put(row);
                }
            }
        }
        buffer.rewind();
        return OnnxTensor.createTensor(environment, buffer, new long[]{a, b, c, d});
    }

    private static float[][][] read3(OnnxTensor tensor) throws OrtException {
        long[] shape = tensor.getInfo().getShape();
        int a = (int) shape[0];
        int b = (int) shape[1];
        int c = (int) shape[2];
        FloatBuffer buffer = tensor.getFloatBuffer();
        float[][][] out = new float[a][b][c];
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                buffer.get(out[i][j], 0, c);
            }
        }
        return out;
    }

    private static float[][][][] read4(OnnxTensor tensor) throws OrtException {
        long[] shape = tensor.getInfo().getShape();
        int a = (int) shape[0];
        int b = (int) shape[1];
        int c = (int) shape[2];
        int d = (int) shape[3];
        FloatBuffer buffer = tensor.getFloatBuffer();
        float[][][][] out = new float[a][b][c][d];
        for (int i = 0; i < a; i++) {
            for (int j = 0; j < b; j++) {
                for (int k = 0; k < c; k++) {
                    buffer.get(out[i][j][k], 0, d);
                }
            }
        }
        return out;
    }

    @Override
    public void close() {
        try {
            encoder.close();
            decoderInit.close();
            decoderStep.close();
        } catch (OrtException ignored) {
            // Best-effort native release.
        }
    }
}
