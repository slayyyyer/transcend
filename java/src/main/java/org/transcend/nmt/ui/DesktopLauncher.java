package org.transcend.nmt.ui;

import javafx.application.Application;

/** Launcher used by packaged desktop builds. */
public final class DesktopLauncher {
    private DesktopLauncher() {
    }

    public static void main(String[] args) {
        Application.launch(TranslatorApp.class, args);
    }
}
