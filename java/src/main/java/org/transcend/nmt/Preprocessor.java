package org.transcend.nmt;

import java.text.Normalizer;
import java.util.regex.Pattern;

/**
 * Text normalization shared with the Python tokenizer trainer.
 *
 * <p>Policy (identical in both runtimes): Unicode NFC, collapse runs of the six
 * ASCII whitespace characters to a single space, then strip those characters
 * from both ends. Non-ASCII spaces such as U+00A0 are preserved as content.
 */
public final class Preprocessor {
    private static final Pattern ASCII_WHITESPACE_RUN = Pattern.compile("[ \\t\\n\\r\\f\\u000B]+");
    private static final String ASCII_WHITESPACE = " \t\n\r\f\u000B";

    private Preprocessor() {
    }

    public static String normalize(String text) {
        if (text == null) {
            throw new IllegalArgumentException("text is null");
        }
        String nfc = Normalizer.normalize(text, Normalizer.Form.NFC);
        String collapsed = ASCII_WHITESPACE_RUN.matcher(nfc).replaceAll(" ");
        int start = 0;
        int end = collapsed.length();
        while (start < end && ASCII_WHITESPACE.indexOf(collapsed.charAt(start)) >= 0) {
            start++;
        }
        while (end > start && ASCII_WHITESPACE.indexOf(collapsed.charAt(end - 1)) >= 0) {
            end--;
        }
        return collapsed.substring(start, end);
    }
}
