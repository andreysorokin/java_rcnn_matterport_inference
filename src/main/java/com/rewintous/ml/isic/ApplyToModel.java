package com.rewintous.ml.isic;

import com.rewintous.ml.isic.model.MaskRCNNModel;
import com.rewintous.ml.isic.utils.dto.DetectionResult;
import org.tensorflow.Graph;
import org.tensorflow.Session;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import static com.rewintous.ml.isic.model.MaskRCNNUtils.maskAsBWImage;

public class ApplyToModel {

    public static void main(String[] args) throws IOException, URISyntaxException {
        MaskRCNNModel model = new MaskRCNNModel();

        try (Graph graph = new Graph()) {
            URL resource = ClassLoader.getSystemClassLoader().getResource("frozen_task_1.pb");
            URL testImageUrl = ClassLoader.getSystemClassLoader().getResource("ISIC_0000016.jpg");
            BufferedImage inputImage = ImageIO.read(testImageUrl);
            graph.importGraphDef(Files.readAllBytes(Paths.get(resource.toURI())), "");
            // Create a config that will dump out device placement of operations.
            try (Session sess = new Session(graph)) {
                List<DetectionResult> detect = model.detect(sess, inputImage);

                int i = 0;

                for (DetectionResult detectionResult : detect) {
                    i++;
                    BufferedImage bufferedImage = maskAsBWImage(detectionResult.getMask());
                    ImageIO.write(bufferedImage, "png", new File("out" + i + ".png"));
                }
            }
        }
    }
}