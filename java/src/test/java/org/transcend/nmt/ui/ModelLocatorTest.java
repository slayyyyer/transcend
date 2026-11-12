package org.transcend.nmt.ui;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ModelLocatorTest {
    @TempDir
    Path tempDir;

    @Test
    void recognisesACompletePackageAndRejectsAnIncompleteOne() throws Exception {
        Path complete = tempDir.resolve("models/scaled_export");
        for (String relative : new String[]{
                "model_config.json", "generation.json", "manifest.json",
                "onnx/encoder.onnx", "onnx/decoder_init.onnx", "onnx/decoder_step.onnx",
                "tokenizer/tokenizer.json", "tokenizer/merges.txt"}) {
            Path file = complete.resolve(relative);
            Files.createDirectories(file.getParent());
            Files.createFile(file);
        }
        assertTrue(ModelLocator.isPackage(complete));

        Path incomplete = tempDir.resolve("models/incomplete");
        Files.createDirectories(incomplete);
        Files.createFile(incomplete.resolve("model_config.json"));
        assertFalse(ModelLocator.isPackage(incomplete));
    }

    @Test
    void findsTheModelBundledWithAWindowsInstall() throws Exception {
        Path packageDir = tempDir.resolve("installed/models/scaled_export");
        copyRequiredFixtureFiles(packageDir);
        Path launcher = tempDir.resolve("installed/Transcend Translator.exe");
        Files.createFile(launcher);

        String previous = System.getProperty("jpackage.app-path");
        try {
            System.setProperty("jpackage.app-path", launcher.toString());
            assertTrue(ModelLocator.findDefault(tempDir.resolve("unrelated")).isPresent());
        } finally {
            if (previous == null) {
                System.clearProperty("jpackage.app-path");
            } else {
                System.setProperty("jpackage.app-path", previous);
            }
        }
    }

    @Test
    void findsTheRepositoryDefaultFromEitherWorkingDirectory() throws Exception {
        Path packageDir = tempDir.resolve("models/scaled_export");
        for (String relative : new String[]{
                "model_config.json", "generation.json", "manifest.json",
                "onnx/encoder.onnx", "onnx/decoder_init.onnx", "onnx/decoder_step.onnx",
                "tokenizer/tokenizer.json", "tokenizer/merges.txt"}) {
            Path file = packageDir.resolve(relative);
            Files.createDirectories(file.getParent());
            Files.createFile(file);
        }

        assertTrue(ModelLocator.findDefault(tempDir).isPresent());
        assertTrue(ModelLocator.findDefault(tempDir.resolve("java")).isPresent());
    }

    private static void copyRequiredFixtureFiles(Path packageDir) throws Exception {
        for (String relative : new String[]{
                "model_config.json", "generation.json", "manifest.json",
                "onnx/encoder.onnx", "onnx/decoder_init.onnx", "onnx/decoder_step.onnx",
                "tokenizer/tokenizer.json", "tokenizer/merges.txt"}) {
            Path file = packageDir.resolve(relative);
            Files.createDirectories(file.getParent());
            Files.createFile(file);
        }
    }
}
