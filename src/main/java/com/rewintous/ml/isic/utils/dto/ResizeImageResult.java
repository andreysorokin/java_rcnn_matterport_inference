package com.rewintous.ml.isic.utils.dto;

import java.awt.*;
import java.awt.image.BufferedImage;

public class ResizeImageResult {
    private BufferedImage resizedImage;
    private Rectangle window;
    private float scale;
    private Padding padding;

    public ResizeImageResult(BufferedImage resizedImage, Rectangle window, float scale, Padding padding) {
        this.resizedImage = resizedImage;
        this.window = window;
        this.scale = scale;
        this.padding = padding;
    }

    public BufferedImage getResizedImage() {
        return resizedImage;
    }

    public Rectangle getWindow() {
        return window;
    }

    public float getScale() {
        return scale;
    }

    public Padding getPadding() {
        return padding;
    }
}
