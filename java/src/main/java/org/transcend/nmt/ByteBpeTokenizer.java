package org.transcend.nmt;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Exact Java reimplementation of the project's byte-level BPE tokenizer.
 *
 * <p>Ids 0..255 are UTF-8 bytes, 256 is the word boundary, then PAD/UNK/BOS/EOS
 * and the two direction tokens, then merge ids. All arithmetic is integral, so
 * token ids are identical to the Python encoder by construction.
 */
public final class ByteBpeTokenizer {
    public static final int SPACE_ID = 256;
    public static final int PAD_ID = 257;
    public static final int UNK_ID = 258;
    public static final int BOS_ID = 259;
    public static final int EOS_ID = 260;
    public static final int LANG_EN_ID = 261;
    public static final int LANG_FR_ID = 262;
    public static final int FIRST_MERGED_ID = 263;

    private final int[] mergeA;
    private final int[] mergeB;
    private final Map<Long, Integer> ranks = new HashMap<>();
    private final Map<Long, Integer> mergeToId = new HashMap<>();
    private final byte[][] pieces;
    private final Map<String, int[]> wordCache = new HashMap<>();
    private final int vocabSize;

    private ByteBpeTokenizer(int[] mergeA, int[] mergeB, byte[][] pieces) {
        this.mergeA = mergeA;
        this.mergeB = mergeB;
        this.pieces = pieces;
        for (int index = 0; index < mergeA.length; index++) {
            long key = key(mergeA[index], mergeB[index]);
            ranks.put(key, index);
            mergeToId.put(key, FIRST_MERGED_ID + index);
        }
        this.vocabSize = FIRST_MERGED_ID + mergeA.length;
    }

    private static long key(int a, int b) {
        return (((long) a) << 32) | (b & 0xffffffffL);
    }

    public static ByteBpeTokenizer load(Path directory) throws IOException {
        Path mergesPath = directory.resolve("merges.txt");
        Path metaPath = directory.resolve("tokenizer.json");
        if (!Files.exists(mergesPath) || !Files.exists(metaPath)) {
            throw new IOException("tokenizer artifacts missing in " + directory);
        }
        JsonObject meta = new Gson().fromJson(Files.readString(metaPath, StandardCharsets.UTF_8), JsonObject.class);
        if (!"transcend-byte-bpe".equals(meta.get("format").getAsString())) {
            throw new IOException("unsupported tokenizer format: " + meta.get("format"));
        }
        List<int[]> merges = new ArrayList<>();
        for (String line : Files.readAllLines(mergesPath, StandardCharsets.UTF_8)) {
            String trimmed = line.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            String[] parts = trimmed.split("\\s+");
            merges.add(new int[]{Integer.parseInt(parts[0]), Integer.parseInt(parts[1])});
        }
        int declared = meta.get("num_merges").getAsInt();
        if (declared != merges.size()) {
            throw new IOException("merge count mismatch: json=" + declared + " file=" + merges.size());
        }
        int[] a = new int[merges.size()];
        int[] b = new int[merges.size()];
        for (int index = 0; index < merges.size(); index++) {
            a[index] = merges.get(index)[0];
            b[index] = merges.get(index)[1];
        }
        byte[][] pieces = new byte[FIRST_MERGED_ID + merges.size()][];
        for (int index = 0; index < 256; index++) {
            pieces[index] = new byte[]{(byte) index};
        }
        for (int index = 0; index < merges.size(); index++) {
            int id = FIRST_MERGED_ID + index;
            byte[] left = pieces[a[index]];
            byte[] right = pieces[b[index]];
            byte[] combined = new byte[left.length + right.length];
            System.arraycopy(left, 0, combined, 0, left.length);
            System.arraycopy(right, 0, combined, left.length, right.length);
            pieces[id] = combined;
        }
        return new ByteBpeTokenizer(a, b, pieces);
    }

    public int vocabSize() {
        return vocabSize;
    }

    public int langId(Language language) {
        return language == Language.FR ? LANG_FR_ID : LANG_EN_ID;
    }

    private int[] encodeWord(String word) {
        int[] cached = wordCache.get(word);
        if (cached != null) {
            return cached;
        }
        byte[] bytes = word.getBytes(StandardCharsets.UTF_8);
        int[] symbols = new int[bytes.length];
        for (int index = 0; index < bytes.length; index++) {
            symbols[index] = bytes[index] & 0xff;
        }
        while (symbols.length > 1) {
            int bestRank = Integer.MAX_VALUE;
            long bestPair = -1;
            for (int index = 0; index < symbols.length - 1; index++) {
                long pair = key(symbols[index], symbols[index + 1]);
                Integer rank = ranks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestPair = pair;
                }
            }
            if (bestPair < 0) {
                break;
            }
            int a = (int) (bestPair >>> 32);
            int b = (int) bestPair;
            int newId = mergeToId.get(bestPair);
            int[] merged = new int[symbols.length];
            int length = 0;
            int index = 0;
            while (index < symbols.length) {
                if (index + 1 < symbols.length && symbols[index] == a && symbols[index + 1] == b) {
                    merged[length++] = newId;
                    index += 2;
                } else {
                    merged[length++] = symbols[index++];
                }
            }
            int[] trimmed = new int[length];
            System.arraycopy(merged, 0, trimmed, 0, length);
            symbols = trimmed;
        }
        if (wordCache.size() < 250_000) {
            wordCache.put(word, symbols);
        }
        return symbols;
    }

    /** Encode text for a direction, optionally forcing the target-language control id. */
    public int[] encode(String text, Language targetLanguage, boolean addEos) {
        String normalized = Preprocessor.normalize(text);
        List<Integer> ids = new ArrayList<>();
        if (targetLanguage != null) {
            ids.add(langId(targetLanguage));
        }
        if (!normalized.isEmpty()) {
            String[] words = normalized.split(" ", -1);
            for (int index = 0; index < words.length; index++) {
                if (index > 0) {
                    ids.add(SPACE_ID);
                }
                for (int id : encodeWord(words[index])) {
                    ids.add(id);
                }
            }
        }
        if (addEos) {
            ids.add(EOS_ID);
        }
        int[] result = new int[ids.size()];
        for (int index = 0; index < result.length; index++) {
            result[index] = ids.get(index);
        }
        return result;
    }

    public String decode(int[] ids) {
        java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream();
        for (int id : ids) {
            if (id < 0) {
                continue;
            }
            if (id < 256) {
                buffer.write(id);
            } else if (id == SPACE_ID) {
                buffer.write(' ');
            } else if (id == PAD_ID || id == UNK_ID || id == BOS_ID || id == EOS_ID
                    || id == LANG_EN_ID || id == LANG_FR_ID) {
                continue;
            } else if (id < pieces.length) {
                buffer.writeBytes(pieces[id]);
            } else {
                throw new IllegalArgumentException("token id outside the trained vocabulary: " + id);
            }
        }
        return new String(buffer.toByteArray(), StandardCharsets.UTF_8);
    }
}
