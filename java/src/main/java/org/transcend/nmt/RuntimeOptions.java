package org.transcend.nmt;

/** Runtime resource and integrity options for {@link Translator#open}. */
public final class RuntimeOptions {
    private final int intraOpThreads;
    private final int maxConcurrency;
    private final boolean verifyChecksums;

    private RuntimeOptions(int intraOpThreads, int maxConcurrency, boolean verifyChecksums) {
        this.intraOpThreads = intraOpThreads;
        this.maxConcurrency = maxConcurrency;
        this.verifyChecksums = verifyChecksums;
    }

    public static RuntimeOptions defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public int intraOpThreads() {
        return intraOpThreads;
    }

    public int maxConcurrency() {
        return maxConcurrency;
    }

    public boolean verifyChecksums() {
        return verifyChecksums;
    }

    public static final class Builder {
        private int intraOpThreads = 0;
        private int maxConcurrency = 4;
        private boolean verifyChecksums = true;

        public Builder intraOpThreads(int value) {
            if (value < 0) {
                throw new IllegalArgumentException("intraOpThreads must be >= 0");
            }
            this.intraOpThreads = value;
            return this;
        }

        public Builder maxConcurrency(int value) {
            if (value < 1) {
                throw new IllegalArgumentException("maxConcurrency must be >= 1");
            }
            this.maxConcurrency = value;
            return this;
        }

        public Builder verifyChecksums(boolean value) {
            this.verifyChecksums = value;
            return this;
        }

        public RuntimeOptions build() {
            return new RuntimeOptions(intraOpThreads, maxConcurrency, verifyChecksums);
        }
    }
}
