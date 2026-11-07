package org.transcend.nmt;

import java.util.List;

/**
 * Structured translation output.
 *
 * @param text              translated text
 * @param modelId           identity of the loaded package (experiment, step, package hash)
 * @param terminationReason {@code eos}, {@code max_length} or {@code empty_input}
 * @param sourceTokens      token count of the encoded source (including control and EOS)
 * @param targetTokens      number of generated tokens, excluding EOS
 * @param warnings          non-fatal notes for the caller
 */
public record TranslationResult(String text, String modelId, String terminationReason,
                                int sourceTokens, int targetTokens, List<String> warnings) {
    public TranslationResult {
        warnings = List.copyOf(warnings);
    }

    /** True when the model stopped naturally rather than hitting the output limit. */
    public boolean completedNaturally() {
        return "eos".equals(terminationReason);
    }
}
