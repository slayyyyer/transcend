package org.transcend.nmt;

/** Raised when an input exceeds the model's supported source length. */
public class InputTooLongException extends IllegalArgumentException {
    private final int tokens;
    private final int limit;

    public InputTooLongException(int tokens, int limit) {
        super("input encodes to " + tokens + " tokens, above the model limit of " + limit
                + "; split the text into shorter segments instead of truncating silently");
        this.tokens = tokens;
        this.limit = limit;
    }

    public int tokens() {
        return tokens;
    }

    public int limit() {
        return limit;
    }
}
