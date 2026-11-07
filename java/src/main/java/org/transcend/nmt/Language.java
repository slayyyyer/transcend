package org.transcend.nmt;

/** Languages supported by the deployed translator. */
public enum Language {
    EN("en"),
    FR("fr");

    private final String code;

    Language(String code) {
        this.code = code;
    }

    public String code() {
        return code;
    }

    public static Language fromCode(String code) {
        if (code == null) {
            throw new IllegalArgumentException("language code is null");
        }
        return switch (code.trim().toLowerCase()) {
            case "en", "eng", "english" -> EN;
            case "fr", "fra", "fre", "french" -> FR;
            default -> throw new IllegalArgumentException("unsupported language: " + code);
        };
    }
}
