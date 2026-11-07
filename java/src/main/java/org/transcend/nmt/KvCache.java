package org.transcend.nmt;

import java.util.ArrayList;
import java.util.List;

/**
 * Decoder KV cache: per-layer self and cross attention tensors shaped
 * {@code [batch][heads][sequence][head_dim]}.
 */
final class KvCache {
    final List<float[][][][]> selfKey;
    final List<float[][][][]> selfValue;
    final List<float[][][][]> crossKey;
    final List<float[][][][]> crossValue;

    KvCache(List<float[][][][]> selfKey, List<float[][][][]> selfValue,
            List<float[][][][]> crossKey, List<float[][][][]> crossValue) {
        this.selfKey = selfKey;
        this.selfValue = selfValue;
        this.crossKey = crossKey;
        this.crossValue = crossValue;
    }

    int layers() {
        return selfKey.size();
    }

    /** Duplicate a batch-1 cache across {@code beam} rows (beam search start). */
    KvCache expand(int beam) {
        if (beam == 1) {
            return this;
        }
        return new KvCache(expandAll(selfKey, beam), expandAll(selfValue, beam),
                expandAll(crossKey, beam), expandAll(crossValue, beam));
    }

    /**
     * Reorder the batch dimension. Beam search uses this to keep each surviving
     * hypothesis attached to the attention state that produced it.
     */
    KvCache reorder(int[] parents) {
        return new KvCache(reorderAll(selfKey, parents), reorderAll(selfValue, parents),
                reorderAll(crossKey, parents), reorderAll(crossValue, parents));
    }

    private static List<float[][][][]> expandAll(List<float[][][][]> tensors, int beam) {
        List<float[][][][]> out = new ArrayList<>(tensors.size());
        for (float[][][][] tensor : tensors) {
            int heads = tensor[0].length;
            int seq = tensor[0][0].length;
            int dim = tensor[0][0][0].length;
            float[][][][] expanded = new float[beam][heads][seq][dim];
            for (int row = 0; row < beam; row++) {
                for (int h = 0; h < heads; h++) {
                    for (int s = 0; s < seq; s++) {
                        System.arraycopy(tensor[0][h][s], 0, expanded[row][h][s], 0, dim);
                    }
                }
            }
            out.add(expanded);
        }
        return out;
    }

    private static List<float[][][][]> reorderAll(List<float[][][][]> tensors, int[] parents) {
        List<float[][][][]> out = new ArrayList<>(tensors.size());
        for (float[][][][] tensor : tensors) {
            int batch = parents.length;
            int heads = tensor[0].length;
            int seq = tensor[0][0].length;
            int dim = tensor[0][0][0].length;
            float[][][][] reordered = new float[batch][heads][seq][dim];
            for (int row = 0; row < batch; row++) {
                int parent = parents[row];
                for (int h = 0; h < heads; h++) {
                    for (int s = 0; s < seq; s++) {
                        System.arraycopy(tensor[parent][h][s], 0, reordered[row][h][s], 0, dim);
                    }
                }
            }
            out.add(reordered);
        }
        return out;
    }
}
