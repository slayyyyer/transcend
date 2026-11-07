package org.transcend.nmt;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PreprocessorTest {
    @Test
    void collapsesAsciiWhitespaceOnly() {
        assertEquals("a b c", Preprocessor.normalize("  a\t b\n\nc  "));
        assertEquals("a\u00a0b", Preprocessor.normalize("a\u00a0b"));
    }

    @Test
    void appliesNfcNormalization() {
        String decomposed = "e\u0301te\u0301";
        assertEquals("été", Preprocessor.normalize(decomposed));
    }

    @Test
    void preservesCaseAndPunctuation() {
        assertEquals("Bonjour, ça va ?", Preprocessor.normalize("Bonjour,   ça va ?"));
    }
}
