package org.transcend.nmt;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Integration test against a real exported package. It is skipped unless
 * {@code -Dtranscend.model.dir=<package>} points at an export directory.
 */
class TranslatorParityTest {
    private static Path modelDir() {
        String property = System.getProperty("transcend.model.dir");
        Assumptions.assumeTrue(property != null && Files.exists(Path.of(property)),
                "set -Dtranscend.model.dir to an exported model package to run this test");
        return Path.of(property);
    }

    @Test
    void javaReproducesPythonGreedyOutput() throws Exception {
        Path directory = modelDir();
        JsonObject vectors = JsonParser.parseString(
                Files.readString(directory.resolve("parity_vectors.json"), StandardCharsets.UTF_8)
        ).getAsJsonObject();
        JsonArray cases = vectors.getAsJsonArray("cases");
        try (Translator translator = Translator.open(directory, RuntimeOptions.defaults())) {
            for (int index = 0; index < cases.size(); index++) {
                JsonObject testCase = cases.get(index).getAsJsonObject();
                // Reproduce the Python run at the budget it was recorded with; the
                // packaged default is longer and would continue past the vector.
                TranslationOptions options = testCase.has("max_new_tokens")
                        ? TranslationOptions.builder()
                                .maxNewTokens(testCase.get("max_new_tokens").getAsInt()).build()
                        : TranslationOptions.defaults();
                TranslationResult result = translator.translate(
                        testCase.get("text").getAsString(),
                        Language.fromCode(testCase.get("source").getAsString()),
                        Language.fromCode(testCase.get("target").getAsString()),
                        options);
                assertEquals(testCase.get("expected_text").getAsString(), result.text(),
                        "text differs for: " + testCase.get("text").getAsString());
                assertEquals(testCase.get("expected_termination").getAsString(), result.terminationReason());
                assertEquals(testCase.getAsJsonArray("expected_ids").size(), result.targetTokens());
            }
        }
    }

    @Test
    void rejectsUnsupportedDirectionsAndEmptyInput() throws Exception {
        Path directory = modelDir();
        try (Translator translator = Translator.open(directory, RuntimeOptions.defaults())) {
            assertThrows(IllegalArgumentException.class,
                    () -> translator.translate("hello", Language.EN, Language.EN, TranslationOptions.defaults()));
            TranslationResult empty = translator.translate("   ", Language.EN, Language.FR,
                    TranslationOptions.defaults());
            assertEquals("", empty.text());
            assertEquals("empty_input", empty.terminationReason());
            assertThrows(IllegalArgumentException.class,
                    () -> translator.translate(null, Language.EN, Language.FR, TranslationOptions.defaults()));
        }
    }

    @Test
    void batchPreservesOrder() throws Exception {
        Path directory = modelDir();
        try (Translator translator = Translator.open(directory, RuntimeOptions.defaults())) {
            var inputs = java.util.List.of("The train leaves at eight.", "Where is the station?");
            var results = translator.translateBatch(inputs, Language.EN, Language.FR,
                    TranslationOptions.defaults());
            assertEquals(inputs.size(), results.size());
            // Batch results must line up with the same inputs translated individually.
            assertEquals(translator.translate(inputs.get(0), Language.EN, Language.FR,
                    TranslationOptions.defaults()).text(), results.get(0).text());
            assertEquals(translator.translate(inputs.get(1), Language.EN, Language.FR,
                    TranslationOptions.defaults()).text(), results.get(1).text());
            assertTrue(results.get(0).sourceTokens() > 0);
        }
    }
}
