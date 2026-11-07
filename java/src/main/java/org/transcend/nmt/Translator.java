package org.transcend.nmt;

import ai.onnxruntime.OrtException;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Semaphore;

/**
 * Offline English&lt;-&gt;French translator backed by the exported ONNX package.
 *
 * <p>Load once, reuse for many calls, and close to release native resources.
 * Instances are thread-safe; concurrent calls are bounded by
 * {@link RuntimeOptions#maxConcurrency()}. Batch calls preserve input order and
 * run items sequentially inside the calling thread.
 */
public final class Translator implements AutoCloseable {
    private final ByteBpeTokenizer tokenizer;
    private final OnnxNmtModel model;
    private final String modelId;
    private final int maxSourceTokens;
    private final int defaultMaxNewTokens;
    private final int defaultBeamSize;
    private final double defaultLengthPenalty;
    private final Set<Integer> suppressed = new HashSet<>();
    private final Semaphore concurrency;
    private volatile boolean closed;

    private Translator(ByteBpeTokenizer tokenizer, OnnxNmtModel model, String modelId,
                       int maxSourceTokens, int defaultMaxNewTokens, int defaultBeamSize,
                       double defaultLengthPenalty, Set<Integer> suppressed, int maxConcurrency) {
        this.tokenizer = tokenizer;
        this.model = model;
        this.modelId = modelId;
        this.maxSourceTokens = maxSourceTokens;
        this.defaultMaxNewTokens = defaultMaxNewTokens;
        this.defaultBeamSize = defaultBeamSize;
        this.defaultLengthPenalty = defaultLengthPenalty;
        this.suppressed.addAll(suppressed);
        this.concurrency = new Semaphore(maxConcurrency, true);
    }

    public static Translator open(Path modelDirectory, RuntimeOptions options) throws IOException {
        RuntimeOptions effective = options == null ? RuntimeOptions.defaults() : options;
        Path configPath = modelDirectory.resolve("model_config.json");
        Path generationPath = modelDirectory.resolve("generation.json");
        Path manifestPath = modelDirectory.resolve("manifest.json");
        for (Path required : List.of(configPath, generationPath, manifestPath)) {
            if (!Files.exists(required)) {
                throw new IOException("missing model artifact: " + required);
            }
        }
        JsonObject config = JsonParser.parseString(Files.readString(configPath, StandardCharsets.UTF_8)).getAsJsonObject();
        JsonObject generation = JsonParser.parseString(Files.readString(generationPath, StandardCharsets.UTF_8)).getAsJsonObject();
        JsonObject manifest = JsonParser.parseString(Files.readString(manifestPath, StandardCharsets.UTF_8)).getAsJsonObject();

        if (effective.verifyChecksums()) {
            verifyChecksums(modelDirectory, manifest);
        }

        ByteBpeTokenizer tokenizer = ByteBpeTokenizer.load(modelDirectory.resolve("tokenizer"));
        JsonObject limits = config.getAsJsonObject("limits");
        int maxSourceTokens = limits.get("max_source_tokens").getAsInt();
        JsonObject checkpoint = config.getAsJsonObject("checkpoint");
        String experiment = manifest.has("experiment") ? manifest.get("experiment").getAsString() : "model";
        String encoderHash = manifest.getAsJsonObject("checksums")
                .get("onnx/encoder.onnx").getAsString().substring(0, 8);
        String modelId = experiment + "@step" + checkpoint.get("step").getAsInt() + "#" + encoderHash;
        Set<Integer> suppressed = new HashSet<>();
        JsonArray suppressedArray = generation.getAsJsonArray("suppressed_token_ids");
        for (int index = 0; index < suppressedArray.size(); index++) {
            suppressed.add(suppressedArray.get(index).getAsInt());
        }
        try {
            OnnxNmtModel model = new OnnxNmtModel(modelDirectory, effective.intraOpThreads());
            return new Translator(tokenizer, model, modelId, maxSourceTokens,
                    generation.get("default_max_new_tokens").getAsInt(),
                    generation.get("default_beam_size").getAsInt(),
                    generation.get("length_penalty").getAsDouble(),
                    suppressed, effective.maxConcurrency());
        } catch (OrtException error) {
            throw new IOException("failed to initialise ONNX Runtime sessions", error);
        }
    }

    private static void verifyChecksums(Path modelDirectory, JsonObject manifest) throws IOException {
        JsonObject checksums = manifest.getAsJsonObject("checksums");
        for (Map.Entry<String, com.google.gson.JsonElement> entry : checksums.entrySet()) {
            String name = entry.getKey();
            // The torch state dict is not used at inference time; skip its large hash.
            if (name.endsWith(".pt")) {
                continue;
            }
            Path file = modelDirectory.resolve(name);
            if (!Files.exists(file)) {
                throw new IOException("artifact listed in manifest is missing: " + name);
            }
            String actual = sha256(file);
            if (!actual.equalsIgnoreCase(entry.getValue().getAsString())) {
                throw new IOException("checksum mismatch for " + name);
            }
        }
    }

    private static String sha256(Path file) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = Files.readAllBytes(file);
            byte[] hashed = digest.digest(bytes);
            StringBuilder builder = new StringBuilder();
            for (byte value : hashed) {
                builder.append(String.format("%02x", value));
            }
            return builder.toString();
        } catch (java.security.NoSuchAlgorithmException error) {
            throw new IOException("SHA-256 unavailable", error);
        }
    }

    public String modelId() {
        return modelId;
    }

    public int maxSourceTokens() {
        return maxSourceTokens;
    }

    public TranslationResult translate(String text, Language source, Language target, TranslationOptions options) {
        ensureOpen();
        ensureDirection(source, target);
        TranslationOptions effective = options == null ? TranslationOptions.defaults() : options;
        String normalized = Preprocessor.normalize(text);
        if (normalized.isEmpty()) {
            return new TranslationResult("", modelId, "empty_input", 0, 0, List.of("empty_input"));
        }
        int[] sourceIds = tokenizer.encode(normalized, target, true);
        if (sourceIds.length > maxSourceTokens) {
            throw new InputTooLongException(sourceIds.length, maxSourceTokens);
        }
        int maxNewTokens = effective.maxNewTokens() == null ? defaultMaxNewTokens : effective.maxNewTokens();
        int beamSize = effective.beamSize() == null ? defaultBeamSize : effective.beamSize();
        double lengthPenalty = effective.lengthPenalty() == null ? defaultLengthPenalty : effective.lengthPenalty();

        concurrency.acquireUninterruptibly();
        try {
            Decoded decoded = generate(sourceIds, maxNewTokens, beamSize, lengthPenalty);
            List<String> warnings = new ArrayList<>();
            if ("max_length".equals(decoded.termination())) {
                warnings.add("output_length_limit_reached: the translation may be truncated");
            }
            String output = tokenizer.decode(decoded.tokens());
            return new TranslationResult(output, modelId, decoded.termination(),
                    sourceIds.length, decoded.tokens().length, warnings);
        } finally {
            concurrency.release();
        }
    }

    public List<TranslationResult> translateBatch(List<String> texts, Language source, Language target,
                                                  TranslationOptions options) {
        ensureOpen();
        ensureDirection(source, target);
        if (texts == null) {
            throw new IllegalArgumentException("texts is null");
        }
        List<TranslationResult> results = new ArrayList<>(texts.size());
        for (String text : texts) {
            results.add(translate(text, source, target, options));
        }
        return results;
    }

    private void ensureDirection(Language source, Language target) {
        if (source == null || target == null) {
            throw new IllegalArgumentException("source and target languages must be set");
        }
        if (source == target) {
            throw new IllegalArgumentException("source and target language are identical: " + source);
        }
        boolean supported = (source == Language.EN && target == Language.FR)
                || (source == Language.FR && target == Language.EN);
        if (!supported) {
            throw new IllegalArgumentException("unsupported translation direction: " + source + "->" + target);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("translator is closed");
        }
    }

    private Decoded generate(int[] sourceIds, int maxNewTokens, int beamSize, double lengthPenalty) {
        try {
            int[] positions = new int[sourceIds.length];
            for (int index = 0; index < positions.length; index++) {
                positions[index] = index;
            }
            float[][][] memory = model.encode(sourceIds, positions);
            if (beamSize <= 1) {
                return greedy(memory, maxNewTokens);
            }
            return beamSearch(memory, maxNewTokens, beamSize, lengthPenalty);
        } catch (OrtException error) {
            throw new IllegalStateException("ONNX Runtime inference failed", error);
        }
    }

    private Decoded greedy(float[][][] memory, int maxNewTokens) throws OrtException {
        OnnxNmtModel.Init init = model.decodeInit(memory);
        float[][] logits = init.logits();
        KvCache cache = init.cache();
        List<Integer> tokens = new ArrayList<>();
        for (int step = 0; step < maxNewTokens; step++) {
            int next = argmax(logits[0]);
            if (next == ByteBpeTokenizer.EOS_ID) {
                return new Decoded(toArray(tokens), "eos", 0.0);
            }
            tokens.add(next);
            if (tokens.size() >= maxNewTokens) {
                break;
            }
            OnnxNmtModel.Step result = model.decodeStep(new int[]{next}, step + 1, cache);
            logits = result.logits();
            cache = result.cache();
        }
        return new Decoded(toArray(tokens), "max_length", 0.0);
    }

    private Decoded beamSearch(float[][][] memory, int maxNewTokens, int beamSize,
                               double lengthPenalty) throws OrtException {
        OnnxNmtModel.Init init = model.decodeInit(memory);
        int vocab = init.logits()[0].length;
        float[][] logits = new float[beamSize][];
        for (int row = 0; row < beamSize; row++) {
            logits[row] = init.logits()[0];
        }
        KvCache cache = init.cache().expand(beamSize);
        double[] scores = new double[beamSize];
        java.util.Arrays.fill(scores, Double.NEGATIVE_INFINITY);
        scores[0] = 0.0;
        List<List<Integer>> hypotheses = new ArrayList<>();
        for (int row = 0; row < beamSize; row++) {
            hypotheses.add(new ArrayList<>());
        }
        Double[] finalScores = new Double[beamSize];

        for (int step = 0; step < maxNewTokens; step++) {
            double[] candidates = new double[beamSize * vocab];
            for (int row = 0; row < beamSize; row++) {
                double[] logProbs = logSoftmax(logits[row]);
                for (int token = 0; token < vocab; token++) {
                    candidates[row * vocab + token] = scores[row] + logProbs[token];
                }
            }
            int[] top = topK(candidates, beamSize);
            double[] newScores = new double[beamSize];
            java.util.Arrays.fill(newScores, Double.NEGATIVE_INFINITY);
            Double[] newFinal = new Double[beamSize];
            List<List<Integer>> newHypotheses = new ArrayList<>();
            int[] parents = new int[beamSize];
            int[] newTokens = new int[beamSize];
            java.util.Arrays.fill(newTokens, ByteBpeTokenizer.EOS_ID);

            for (int rank = 0; rank < beamSize; rank++) {
                int flat = top[rank];
                int parent = flat / vocab;
                int token = flat % vocab;
                parents[rank] = parent;
                List<Integer> parentTokens = hypotheses.get(parent);
                if (finalScores[parent] != null) {
                    newHypotheses.add(parentTokens);
                    newFinal[rank] = finalScores[parent];
                    continue;
                }
                if (token == ByteBpeTokenizer.EOS_ID) {
                    newHypotheses.add(parentTokens);
                    double length = parentTokens.size() + 1.0;
                    double penalty = Math.pow((5.0 + length) / 6.0, lengthPenalty);
                    newFinal[rank] = candidates[flat] / penalty;
                } else {
                    List<Integer> extended = new ArrayList<>(parentTokens);
                    extended.add(token);
                    newHypotheses.add(extended);
                    newScores[rank] = candidates[flat];
                    newTokens[rank] = token;
                }
            }

            hypotheses = newHypotheses;
            finalScores = newFinal;
            scores = newScores;
            boolean allFinished = true;
            for (Double value : finalScores) {
                if (value == null) {
                    allFinished = false;
                    break;
                }
            }
            if (allFinished) {
                break;
            }
            cache = cache.reorder(parents);
            OnnxNmtModel.Step result = model.decodeStep(newTokens, step + 1, cache);
            logits = result.logits();
            cache = result.cache();
        }

        int bestIndex = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int row = 0; row < beamSize; row++) {
            if (finalScores[row] != null && finalScores[row] > bestScore) {
                bestScore = finalScores[row];
                bestIndex = row;
            }
        }
        if (bestIndex >= 0) {
            return new Decoded(toArray(hypotheses.get(bestIndex)), "eos", bestScore);
        }
        for (int row = 0; row < beamSize; row++) {
            if (scores[row] > bestScore) {
                bestScore = scores[row];
                bestIndex = row;
            }
        }
        if (bestIndex < 0) {
            bestIndex = 0;
        }
        return new Decoded(toArray(hypotheses.get(bestIndex)), "max_length", bestScore);
    }

    private int argmax(float[] logits) {
        int best = -1;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int token = 0; token < logits.length; token++) {
            if (suppressed.contains(token)) {
                continue;
            }
            if (logits[token] > bestValue) {
                bestValue = logits[token];
                best = token;
            }
        }
        if (best < 0) {
            throw new IllegalStateException("all output tokens were suppressed");
        }
        return best;
    }

    private double[] logSoftmax(float[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (int token = 0; token < logits.length; token++) {
            if (suppressed.contains(token)) {
                continue;
            }
            if (logits[token] > max) {
                max = logits[token];
            }
        }
        double[] result = new double[logits.length];
        double sum = 0.0;
        for (int token = 0; token < logits.length; token++) {
            if (suppressed.contains(token)) {
                result[token] = Double.NEGATIVE_INFINITY;
                continue;
            }
            double value = Math.exp(logits[token] - max);
            result[token] = value;
            sum += value;
        }
        for (int token = 0; token < logits.length; token++) {
            if (result[token] != Double.NEGATIVE_INFINITY) {
                result[token] = Math.log(result[token] / sum);
            }
        }
        return result;
    }

    private static int[] topK(double[] values, int k) {
        int[] bestIndices = new int[k];
        double[] bestValues = new double[k];
        java.util.Arrays.fill(bestValues, Double.NEGATIVE_INFINITY);
        java.util.Arrays.fill(bestIndices, -1);
        for (int index = 0; index < values.length; index++) {
            double value = values[index];
            if (value <= bestValues[k - 1]) {
                continue;
            }
            int position = k - 1;
            while (position > 0 && bestValues[position - 1] < value) {
                bestValues[position] = bestValues[position - 1];
                bestIndices[position] = bestIndices[position - 1];
                position--;
            }
            bestValues[position] = value;
            bestIndices[position] = index;
        }
        return bestIndices;
    }

    private static int[] toArray(List<Integer> values) {
        int[] out = new int[values.size()];
        for (int index = 0; index < out.length; index++) {
            out[index] = values.get(index);
        }
        return out;
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        closed = true;
        model.close();
    }

    private record Decoded(int[] tokens, String termination, double score) {
    }
}
