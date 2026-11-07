package org.transcend.nmt;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CacheReorderTest {
    private static float[][][][] cacheWithValue(float value, int batch) {
        float[][][][] tensor = new float[batch][2][3][4];
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < 2; h++) {
                for (int s = 0; s < 3; s++) {
                    for (int d = 0; d < 4; d++) {
                        tensor[b][h][s][d] = value + b;
                    }
                }
            }
        }
        return tensor;
    }

    private static List<float[][][][]> wrap(float[][][][] tensor) {
        return List.<float[][][][]>of(tensor);
    }

    @Test
    void reorderMovesAttentionStateWithItsBeam() {
        KvCache cache = new KvCache(
                wrap(cacheWithValue(10f, 2)),
                wrap(cacheWithValue(20f, 2)),
                wrap(cacheWithValue(30f, 2)),
                wrap(cacheWithValue(40f, 2)));
        KvCache reordered = cache.reorder(new int[]{1, 0});
        assertEquals(11f, reordered.selfKey.get(0)[0][0][0][0], 1e-6);
        assertEquals(10f, reordered.selfKey.get(0)[1][0][0][0], 1e-6);
        assertEquals(21f, reordered.selfValue.get(0)[0][0][0][0], 1e-6);
        assertEquals(31f, reordered.crossKey.get(0)[0][0][0][0], 1e-6);
    }

    @Test
    void expandDuplicatesBatchOne() {
        float[][][][] key = cacheWithValue(5f, 1);
        KvCache cache = new KvCache(wrap(key), wrap(key), wrap(key), wrap(key));
        KvCache expanded = cache.expand(3);
        for (int row = 0; row < 3; row++) {
            assertEquals(5f, expanded.selfKey.get(0)[row][0][0][0], 1e-6);
        }
    }
}
