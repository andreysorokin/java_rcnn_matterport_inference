package com.rewintous.ml.isic.model;

import com.rewintous.ml.isic.utils.ImageUtils;
import com.rewintous.ml.isic.utils.dto.DetectionResult;
import com.rewintous.ml.isic.utils.dto.FloatBox;
import com.rewintous.ml.isic.utils.dto.ImageShape;
import com.rewintous.ml.isic.utils.dto.ResizeImageResult;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.*;
import java.util.List;

import static java.lang.Math.floor;
import static java.lang.Math.max;

public class MaskRCNNModel {
    private final MaskRCNNConfig maskRCNNConfig;

    public MaskRCNNModel() {
        this(new MaskRCNNConfig());
    }

    public MaskRCNNModel(MaskRCNNConfig maskRCNNConfig) {
        this.maskRCNNConfig = maskRCNNConfig;
    }

    public List<FloatBox> getAnchors(int height, int width) {
        return null;
    }

    public List<DetectionResult> detect(Session sess, BufferedImage bufferedImage) {
        ResizeImageResult resizeImageResult = ImageUtils.resize_image(bufferedImage,
                maskRCNNConfig.getImageMinDim(),
                maskRCNNConfig.getImageMaxDim(),
                maskRCNNConfig.getImageMinScale(),
                maskRCNNConfig.getImageResizeMode()
        );

        /**
         * 0 = {Tensor} Tensor("input_image:0", shape=(?, ?, ?, 3), dtype=float32)
         * 1 = {Tensor} Tensor("input_image_meta:0", shape=(?, 14), dtype=float32)
         * 2 = {Tensor} Tensor("input_anchors:0", shape=(?, ?, 4), dtype=float32)
         */

        //Component 0 - input image
        INDArray moldImage = moldImage(resizeImageResult.getResizedImage());

        int width = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();
        int channels = 3;

        //Component 1 - metadata
        long[] shape = moldImage.shape();
        int[] activeClassIds = new int[maskRCNNConfig.getNumClasses()]; //zeros
        int inferencedWidth = (int) shape[0];
        int inferencedHeight = (int) shape[1];
        float[] meta = MaskRCNNUtils.compose_image_meta(0,
                new int[]{height, width, channels},
                new int[]{inferencedWidth, inferencedHeight, (int) shape[2]},
                new FloatBox(resizeImageResult.getWindow()),
                resizeImageResult.getScale(),
                activeClassIds
        );

        //Component 2 - normalized anchors
        List<FloatBox> floatBoxes = generateAnchors(inferencedHeight, inferencedWidth);

        /* Tensors and shapes
         * 0 = {Tensor} Tensor("input_image:0", shape=(1, H, W, 3), dtype=float32)
         * 1 = {Tensor} Tensor("input_image_meta:0", shape=(1 14), dtype=float32)
         * 2 = {Tensor} Tensor("input_anchors:0", shape=(?, ?, 4), dtype=float32) <class 'tuple'>: (1, 147312, 4) (y1, x1, y2, x2)
         */

        /*
           Output names <class 'list'>: [
            'mrcnn_detection',  <class 'tuple'>: (1, 400, 6)
            'mrcnn_class',      <class 'tuple'>: (1, 2000, 2)
            'mrcnn_bbox',       <class 'tuple'>: (1, 2000, 2, 4)
            'mrcnn_mask',       <class 'tuple'>: (1, 400, 28, 28, 2)
            'ROI',              <class 'tuple'>: (1, 2000, 4)
            'rpn_class',        <class 'tuple'>: (1, 147312, 2)
            'rpn_bbox']         <class 'tuple'>: (1, 147312, 4)
         */

        float[][][][] inputImage = new float[1][inferencedWidth][inferencedHeight][3];

        for (int h = 0; h < shape[0]; h++) {
            for (int w = 0; w < shape[1]; w++) {
                inputImage[0][h][w][0] = (float) moldImage.getDouble(h, w, 0);
                inputImage[0][h][w][1] = (float) moldImage.getDouble(h, w, 1);
                inputImage[0][h][w][2] = (float) moldImage.getDouble(h, w, 2);
            }
        }

        Tensor<Float> inputImageTensor = Tensors.create(inputImage);

        float[][] inputImageMeta = new float[1][14];

        for (int i = 0; i < meta.length; i++) {
            inputImageMeta[0][i] = meta[i];
        }

        Tensor<Float> metaTensor = Tensors.create(inputImageMeta);

        float[][][] boxes = new float[1][floatBoxes.size()][4];

        int boxId = 0;
        for (FloatBox floatBox : floatBoxes) {
            boxes[0][boxId][0] = floatBox.getY1();
            boxes[0][boxId][1] = floatBox.getX1();
            boxes[0][boxId][2] = floatBox.getY2();
            boxes[0][boxId][3] = floatBox.getX2();
            boxId++;
        }

        Tensor<Float> anchorsTensor = Tensors.create(boxes);

        List<Tensor<?>> result = sess.runner()
                .feed("input_image", inputImageTensor)
                .feed("input_image_meta", metaTensor)
                .feed("input_anchors", anchorsTensor)
                .fetch("mrcnn_detection/Reshape_1:0")
                .fetch("mrcnn_mask/Reshape_1:0")
                .run();

        float[][][] detectionBuffer = new float[1][400][6];
        float[][][][][] maskBuffer = new float[1][400][28][28][2];

        result.get(0).expect(Float.class).copyTo(detectionBuffer);
        result.get(1).expect(Float.class).copyTo(maskBuffer);

        List<DetectionResult> results = unmoldDetections(detectionBuffer[0], maskBuffer[0], width, height, inferencedWidth, inferencedHeight,
                new FloatBox(resizeImageResult.getWindow()));

        return results;
    }

    protected INDArray moldImage(BufferedImage bufferedImage) {
        INDArray indArray = ImageUtils.asMatrix(bufferedImage);
        //todo replace with matrix operations
        for (int h = 0; h < indArray.shape()[0]; h++) {
            for (int w = 0; w < indArray.shape()[1]; w++) {
                indArray.putScalar(h, w, 0, indArray.getDouble(h, w, 0) - maskRCNNConfig.getMeanPixel()[0]);
                indArray.putScalar(h, w, 1, indArray.getDouble(h, w, 1) - maskRCNNConfig.getMeanPixel()[1]);
                indArray.putScalar(h, w, 2, indArray.getDouble(h, w, 2) - maskRCNNConfig.getMeanPixel()[2]);
            }
        }

        return indArray;
    }

    private List<FloatBox> generateAnchors(int width, int height) {
        List<ImageShape> shapes = computeBackboneShapes(width, height);

        int[] widths = new int[shapes.size()];
        int[] heights = new int[shapes.size()];

        Iterator<ImageShape> iterator = shapes.iterator();
        for (int i = 0; i < widths.length; i++) {
            ImageShape next = iterator.next();
            widths[i] = next.getWidth();
            heights[i] = next.getHeight();
        }

        List<FloatBox> floatBoxes = MaskRCNNUtils.generatePyramidAnchors(
                maskRCNNConfig.getRpnAnchorScales(),
                maskRCNNConfig.getRpnAnchorRatios(),
                widths, heights,
                maskRCNNConfig.getBackboneStrides(),
                (float) maskRCNNConfig.getRpnAnchorStride());

        floatBoxes.forEach(fb -> fb.normalize(width, height));

        return floatBoxes;
    }

    private List<ImageShape> computeBackboneShapes(int width, int height) {
        List<ImageShape> shapes = new LinkedList<>();

        int[] backboneStrides = maskRCNNConfig.getBackboneStrides();
        for (int i = 0; i < backboneStrides.length; i++) {
            int backboneStride = backboneStrides[i];
            shapes.add(new ImageShape(width / backboneStride, height / backboneStride));
        }
        return shapes;
    }

    /**
     * @param detectionBuffer [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
     * @param maskBuffer      [N, height, width, num_classes]
     * @param originalWidth   original image width
     * @param originalHeight  original image height
     * @param width           molded image width
     * @param height          molded image height
     * @param window          window of molded image on original image
     * @return Returns:
     * boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
     * class_ids: [N] Integer class IDs for each bounding box
     * scores: [N] Float probability scores of the class_id
     * masks: [height, width, num_instances] Instance masks
     */
    private List<DetectionResult> unmoldDetections(float[][] detectionBuffer, float[][][][] maskBuffer,
                                                   int originalWidth, int originalHeight, int width, int height, FloatBox window) {

        long numDetections = Arrays.stream(detectionBuffer).filter(buff -> buff[4] > 0).count();

        if (numDetections == 0) {
            return Collections.emptyList();
        }

        window.normalize(originalWidth, originalHeight);

        float wh = window.getY2() - window.getY1();
        float ww = window.getX2() - window.getX1();

        List<DetectionResult> results = new LinkedList<>();

        for (int i = 0; i < numDetections; i++) {
            float[] detectionArray = detectionBuffer[i];
            FloatBox floatBox =
                    new FloatBox(
                            (detectionArray[1] - window.getX1()) / ww, (detectionArray[0] - -window.getY1()) / wh,
                            (detectionArray[3] - -window.getX1()) / ww, (detectionArray[2] - -window.getY1()) / wh
                    );
            floatBox.denormalize(originalWidth, originalHeight);

            if (floatBox.getArea() <= 0) {
                continue;
            }

            int classId = (int) detectionArray[4];
            float score = detectionArray[5];

            float[][][] detectionMask = maskBuffer[i];
            //normally (h,w) mask
            //mask height, width

            int maskHeight = detectionMask.length;
            int maskWidth = detectionMask[0].length;

            float[][] mask = new float[maskHeight][maskWidth];

            for (int h = 0; h < maskHeight; h++) {
                for (int w = 0; w < maskWidth; w++) {
                    mask[h][w] = detectionMask[h][w][classId];
                }
            }

            boolean[][] booleanMask = unmold_mask(mask, floatBox, originalWidth, originalHeight);
            results.add(new DetectionResult(floatBox, classId, score, booleanMask));
        }

        return results;
    }

    private boolean[][] unmold_mask(float[][] mask, FloatBox boundaryBox, int imageWidth, int imageHeight) {
        int width = mask[0].length;
        int height = mask.length;

        //note resize is done via resizing grayscale image with byte discretization, use numpy.resize or native resize
        // if needed
        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster raster = bufferedImage.getRaster();

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                raster.setPixel(w,h, new int[]{(int)(0xFF * mask[h][w])});
            }
        }

        int scaledWidth = (int) boundaryBox.getWidth();
        int scaledHeight = (int) boundaryBox.getHeight();
        BufferedImage scaledImage =
                new BufferedImage(scaledWidth, scaledHeight, BufferedImage.TYPE_BYTE_GRAY);

        Graphics2D g2d = scaledImage.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING,
                RenderingHints.VALUE_RENDER_QUALITY);
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);


        g2d.drawImage(bufferedImage, 0, 0, scaledWidth, scaledHeight, null);
        g2d.dispose();
        WritableRaster scaledRaster = scaledImage.getRaster();

        boolean[][] outMask = new boolean[imageHeight][imageWidth];
        for (int h = 0; h < imageHeight; h++) {
            for (int w = 0; w < imageWidth; w++) {
                int x = (int) floor(w - boundaryBox.getX1());
                int y = (int) floor(h - boundaryBox.getY1());
                if (0 <= x && x < scaledWidth && 0 <= y && y < scaledHeight /*&& */) {
                    if (scaledRaster.getPixel(x, y, (int[])null)[0] > 0x7F) {
                        outMask[h][w] = true;
                    }
                }
            }
        }

        return outMask;
    }
}
