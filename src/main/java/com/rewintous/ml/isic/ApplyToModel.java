package com.rewintous.ml.isic;

import com.rewintous.ml.isic.model.MaskRCNNModel;
import org.tensorflow.Graph;
import org.tensorflow.Session;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ApplyToModel {

//    TensorFlow tensorFlow;

    public static void main(String[] args) throws IOException, URISyntaxException {
        MaskRCNNModel model = new MaskRCNNModel();

        try (Graph graph = new Graph()) {
            URL resource = ClassLoader.getSystemClassLoader().getResource("frozen_task_1.pb");
            URL testImageUrl = ClassLoader.getSystemClassLoader().getResource("ISIC_0000016.jpg");
            BufferedImage inputImage = ImageIO.read(testImageUrl);
            graph.importGraphDef(Files.readAllBytes(Paths.get(resource.toURI())), "");
            // Create a config that will dump out device placement of operations.
            try (Session sess = new Session(graph)) {
                model.detect(sess, inputImage);
/*
                String pathname = "table246.jpg";
                BufferedImage read = ImageIO.read(new File(pathname));
                BufferedImage resized = resizeImageWithHint(read, width, height, BufferedImage.TYPE_INT_RGB);
                float[][][][] inputFloat = convertImageToArray4(resized);


                try (Tensor<Float> in = Tensors.create(inputFloat);
                     Tensor<Float> out = sess.runner().feed("input", in).fetch("output").run().get(0).expect(Float.class)) {

                    float[][][][] outData = new float[1][gridWidth][gridHeight][blockSize];
                    out.copyTo(outData);

                    TensorFlowYoloDetector tensorFlowYoloDetector = new TensorFlowYoloDetector();
                    List<Recognition> recognitions = tensorFlowYoloDetector.recognizeImage(outData, read.getWidth(), read.getHeight());
                    System.out.println("recognitions = " + recognitions);
                    BufferedImage bufferedImage = drawProposals(read, recognitions);
                    ImageIO.write(bufferedImage, "png", new File("proposals-" + pathname + ".png"));

                }
*/
            }
        }
    }



/*    private static BufferedImage drawProposals(BufferedImage originalImage, Iterable<Recognition> recognitions) {

        int width = originalImage.getWidth();
        int height = originalImage.getHeight();
        BufferedImage resizedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(originalImage, 0, 0, width, height, null);

        int cIdx = 0;
        for (Recognition recognition : recognitions) {
            Color color = Color.decode(mColors[cIdx]);
            g.setColor(color);

            int x = Math.round(recognition.getLocation().getX() * width);
            int y = Math.round(recognition.getLocation().getY() * height);
            int w = Math.round(recognition.getLocation().getW() * width);
            int h = Math.round(recognition.getLocation().getH() * height);
            g.drawRect(x, y, w, h);

            cIdx = (cIdx + 1) % mColors.length;
        }


        g.dispose();
        g.setComposite(AlphaComposite.Src);

        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setRenderingHint(RenderingHints.KEY_RENDERING,
                RenderingHints.VALUE_RENDER_QUALITY);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);

        return resizedImage;
    }

    private static float[][][][] convertImageToArray4(BufferedImage bf) {
        int width = bf.getWidth();
        int height = bf.getHeight();
        int[] data = new int[width * height];
        bf.getRGB(0, 0, width, height, data, 0, width);
        float[][][][] rgbArray = new float[1][width][height][3];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int color = data[i * width + j];
                /// 0, y, x, plane
                rgbArray[0][i][j][0] = ((color >> 16) & 0x000000FF) / 255.0f;
                rgbArray[0][i][j][1] = ((color >> 8) & 0x000000FF) / 255.0f;
                rgbArray[0][i][j][2] = ((color) & 0x000000FF) / 255.0f;
            }
        }
        return rgbArray;
    }

    private static int gridHeight = 13;
    private static int gridWidth = 13;
    private static int blockSize = 30;


    private final static String[] mColors = {
            "#39add1", // light blue
            "#3079ab", // dark blue
            "#c25975", // mauve
            "#e15258", // red
            "#f9845b", // orange
            "#838cc7", // lavender
            "#7d669e", // purple
            "#53bbb4", // aqua
            "#51b46d", // green
            "#e0ab18", // mustard
            "#637a91", // dark gray
            "#f092b0", // pink
            "#b7c0c7"  // light gray
    };*/
}
