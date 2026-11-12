package org.transcend.nmt.ui;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/** Locates an exported model package without making the window depend on the shell's cwd. */
public final class ModelLocator {
    private static final List<String> REQUIRED = List.of(
            "model_config.json",
            "generation.json",
            "manifest.json",
            "onnx/encoder.onnx",
            "onnx/decoder_init.onnx",
            "onnx/decoder_step.onnx",
            "tokenizer/tokenizer.json",
            "tokenizer/merges.txt");

    private ModelLocator() {
    }

    /** Looks for a package near the project root, in the environment, or in a system property. */
    public static Optional<Path> findDefault(Path workingDirectory) {
        List<Path> candidates = new ArrayList<>();
        addIfSet(candidates, System.getProperty("transcend.model.dir"));
        addIfSet(candidates, System.getenv("TRANSCEND_MODEL_DIR"));
        addInstalledPackage(candidates);

        Path root = workingDirectory.toAbsolutePath().normalize();
        candidates.add(root.resolve("models/scaled_export"));
        candidates.add(root.resolve("../models/scaled_export"));
        candidates.add(root.resolve("../../models/scaled_export"));

        for (Path candidate : candidates) {
            Path normalized = candidate.toAbsolutePath().normalize();
            if (isPackage(normalized)) {
                return Optional.of(normalized);
            }
        }
        return Optional.empty();
    }

    public static boolean isPackage(Path directory) {
        if (directory == null || !Files.isDirectory(directory)) {
            return false;
        }
        for (String required : REQUIRED) {
            if (!Files.isRegularFile(directory.resolve(required))) {
                return false;
            }
        }
        return true;
    }

    private static void addIfSet(List<Path> candidates, String value) {
        if (value != null && !value.isBlank()) {
            candidates.add(Path.of(value));
        }
    }

    private static void addInstalledPackage(List<Path> candidates) {
        String appPath = System.getProperty("jpackage.app-path");
        if (appPath == null || appPath.isBlank()) {
            return;
        }
        try {
            Path installRoot = Path.of(appPath).toAbsolutePath().normalize().getParent();
            if (installRoot != null) {
                candidates.add(installRoot.resolve("models/scaled_export"));
            }
        } catch (RuntimeException ignored) {
            // A malformed launcher property should not prevent other search paths.
        }
    }
}
