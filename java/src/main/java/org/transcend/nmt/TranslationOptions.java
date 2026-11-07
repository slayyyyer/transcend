package org.transcend.nmt;

/** Per-call decoding options. Unset values fall back to the packaged defaults. */
public final class TranslationOptions {
    private final Integer maxNewTokens;
    private final Integer beamSize;
    private final Double lengthPenalty;

    private TranslationOptions(Integer maxNewTokens, Integer beamSize, Double lengthPenalty) {
        this.maxNewTokens = maxNewTokens;
        this.beamSize = beamSize;
        this.lengthPenalty = lengthPenalty;
    }

    public static TranslationOptions defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Integer maxNewTokens() {
        return maxNewTokens;
    }

    public Integer beamSize() {
        return beamSize;
    }

    public Double lengthPenalty() {
        return lengthPenalty;
    }

    public static final class Builder {
        private Integer maxNewTokens;
        private Integer beamSize;
        private Double lengthPenalty;

        public Builder maxNewTokens(int value) {
            if (value < 1) {
                throw new IllegalArgumentException("maxNewTokens must be >= 1");
            }
            this.maxNewTokens = value;
            return this;
        }

        public Builder beamSize(int value) {
            if (value < 1) {
                throw new IllegalArgumentException("beamSize must be >= 1");
            }
            this.beamSize = value;
            return this;
        }

        public Builder lengthPenalty(double value) {
            this.lengthPenalty = value;
            return this;
        }

        public TranslationOptions build() {
            return new TranslationOptions(maxNewTokens, beamSize, lengthPenalty);
        }
    }
}
