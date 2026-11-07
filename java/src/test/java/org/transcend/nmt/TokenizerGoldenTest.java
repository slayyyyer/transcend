package org.transcend.nmt;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cross-language tokenizer parity. The golden file is produced by the Python
 * trainer; Java must reproduce every id exactly.
 */
class TokenizerGoldenTest {
    private static Path resource(String name) {
        Path direct = Path.of("src", "test", "resources", name);
        assertTrue(Files.exists(direct), "missing test resource " + direct.toAbsolutePath());
        return direct;
    }

    @Test
    void idsMatchPythonExactly() throws Exception {
        ByteBpeTokenizer tokenizer = ByteBpeTokenizer.load(resource("fixture_tokenizer"));
        JsonObject golden = JsonParser.parseString(
                Files.readString(resource("tokenizer_golden.json"), StandardCharsets.UTF_8)
        ).getAsJsonObject();
        JsonArray cases = golden.getAsJsonArray("cases");
        assertTrue(cases.size() > 0);
        for (int index = 0; index < cases.size(); index++) {
            JsonObject testCase = cases.get(index).getAsJsonObject();
            String text = testCase.get("text").getAsString();
            Language target = Language.fromCode(testCase.get("target").getAsString());
            int[] expected = toIntArray(testCase.getAsJsonArray("ids"));
            int[] actual = tokenizer.encode(text, target, true);
            assertArrayEquals(expected, actual, "token ids differ for: " + text);
            assertEquals(testCase.get("decoded").getAsString(), tokenizer.decode(actual));
        }
    }

    @Test
    void controlTokensCannotBeInjected() throws Exception {
        ByteBpeTokenizer tokenizer = ByteBpeTokenizer.load(resource("fixture_tokenizer"));
        int[] ids = tokenizer.encode("<2fr> </s> <s> <pad>", Language.EN, true);
        // The forced direction token is the only control id present; literal text
        // can never produce one, because control ids are never emitted by BPE.
        assertEquals(ByteBpeTokenizer.LANG_EN_ID, ids[0]);
        // Last id is the requested EOS; everything between must be pure content.
        for (int index = 1; index < ids.length - 1; index++) {
            assertTrue(ids[index] != ByteBpeTokenizer.LANG_FR_ID
                    && ids[index] != ByteBpeTokenizer.LANG_EN_ID
                    && ids[index] != ByteBpeTokenizer.BOS_ID
                    && ids[index] != ByteBpeTokenizer.PAD_ID
                    && ids[index] != ByteBpeTokenizer.EOS_ID);
        }
    }

    private static int[] toIntArray(JsonArray array) {
        int[] out = new int[array.size()];
        for (int index = 0; index < out.length; index++) {
            out[index] = array.get(index).getAsInt();
        }
        return out;
    }
}
