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

import java.awt.image.BufferedImage;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

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

    public DetectionResult detect(Session sess, BufferedImage bufferedImage) {
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
        int height = bufferedImage.getWidth();
        int channels = 3;

        //Component 1 - metadata
        long[] shape = moldImage.shape();
        int[] activeClassIds = new int[maskRCNNConfig.getNumClasses()]; //zeros
        float [] meta = MaskRCNNUtils.compose_image_meta(0,
                new int[]{height, width, channels},
                new int[]{(int) shape[0], (int) shape[1], (int) shape[2]},
                new FloatBox(resizeImageResult.getWindow()),
                resizeImageResult.getScale(),
                activeClassIds
        );

        //Component 2 - normalized anchors
        List<FloatBox> floatBoxes = generateAnchors(width, height);

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

        float [][][][] inputImage = new float[1][height][width][3];

        for (int h=0;h<height; h++) {
            for (int w=0;w<width;w++) {
                inputImage[0][h][w][0] = (float) moldImage.getDouble(h, w, 0);
                inputImage[0][h][w][1] = (float) moldImage.getDouble(h, w, 1);
                inputImage[0][h][w][2] = (float) moldImage.getDouble(h, w, 2);
            }
        }

        Tensor<Float> inputImageTensor = Tensors.create(inputImage);

        float [][] inputImageMeta = new float[1][14];

        for (int i = 0; i < meta.length; i++) {
            inputImageMeta[0][i] = meta[i];
        }

        Tensor<Float> metaTensor = Tensors.create(inputImageMeta);

        float [][][] boxes = new float[1][floatBoxes.size()][4];

        int boxId = 0;
        for (FloatBox floatBox : floatBoxes) {
            boxes[0][boxId][0] = floatBox.getY1();
            boxes[0][boxId][1] = floatBox.getX1();
            boxes[0][boxId][2] = floatBox.getY2();
            boxes[0][boxId][3] = floatBox.getX2();
            boxId++;
        }

        Tensor<Float> anchorsTensor = Tensors.create(boxes);


        /**
         * outputs = {list} <class 'list'>: [<tf.Tensor 'mrcnn_detection/Reshape_1:0' shape=(1, 400, 6) dtype=float32>, <tf.Tensor 'mrcnn_class/Reshape_1:0' shape=(?, 2000, 2) dtype=float32>, <tf.Tensor 'mrcnn_bbox/Reshape:0' shape=(?, 2000, 2, 4) dtype=float32>, <tf.Tensor 'mrcnn_mask/Reshape_1:0' shape=(?, 400, 28, 28, 2) dtype=float32>, <tf.Tensor 'ROI/packed_2:0' shape=(1, ?, ?) dtype=float32>, <tf.Tensor 'rpn_class/concat:0' shape=(?, ?, 2) dtype=float32>, <tf.Tensor 'rpn_bbox/concat:0' shape=(?, ?, 4) dtype=float32>]
         *  0 = {Tensor} Tensor("mrcnn_detection/Reshape_1:0", shape=(1, 400, 6), dtype=float32)
         *  1 = {Tensor} Tensor("mrcnn_class/Reshape_1:0", shape=(?, 2000, 2), dtype=float32)
         *  2 = {Tensor} Tensor("mrcnn_bbox/Reshape:0", shape=(?, 2000, 2, 4), dtype=float32)
         *  3 = {Tensor} Tensor("mrcnn_mask/Reshape_1:0", shape=(?, 400, 28, 28, 2), dtype=float32)
         *  4 = {Tensor} Tensor("ROI/packed_2:0", shape=(1, ?, ?), dtype=float32)
         *  5 = {Tensor} Tensor("rpn_class/concat:0", shape=(?, ?, 2), dtype=float32)
         *  6 = {Tensor} Tensor("rpn_bbox/concat:0", shape=(?, ?, 4), dtype=float32)
         */

        /*
          fixme
          https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py#L1905-L1919
          Exception in thread "main" java.lang.IllegalArgumentException: Incompatible shapes: [1,44,44,256] vs. [1,43,43,256]

	 [[Node: fpn_p4add/add = Add[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/device:CPU:0"](fpn_p5upsampled/ResizeNearestNeighbor, fpn_c4p4/BiasAdd)]]
	at org.tensorflow.Session.run(Native Method)
	at org.tensorflow.Session.access$100(Session.java:48)
	at org.tensorflow.Session$Runner.runHelper(Session.java:298)
	at org.tensorflow.Session$Runner.run(Session.java:248)
	at com.rewintous.ml.isic.model.MaskRCNNModel.detect(MaskRCNNModel.java:137)
	at com.rewintous.ml.isic.ApplyToModel.main(ApplyToModel.java:29)
         */

        List<Tensor<?>> result = sess.runner()
                .feed("input_image", inputImageTensor)
                .feed("input_image_meta", metaTensor)
                .feed("input_anchors", anchorsTensor)
//                .fetch("mrcnn_detection")
                .fetch("mrcnn_detection/Reshape_1:0")
                .fetch("mrcnn_mask/Reshape_1:0")
                .run();


        return null;
    }

    INDArray moldImage(BufferedImage bufferedImage) {
        INDArray indArray = ImageUtils.asMatrix(bufferedImage);
        //todo replace with matrix operations
        for (int h=0; h<indArray.shape()[0]; h++) {
            for (int w=0; w<indArray.shape()[1]; w++) {
                indArray.putScalar(h, w, 0, indArray.getDouble(h,w,0) - maskRCNNConfig.getMeanPixel()[0]);
                indArray.putScalar(h, w, 1, indArray.getDouble(h,w,1) - maskRCNNConfig.getMeanPixel()[1]);
                indArray.putScalar(h, w, 2, indArray.getDouble(h,w,2) - maskRCNNConfig.getMeanPixel()[2]);
            }
        }

        return indArray;
    }

    List<FloatBox> generateAnchors(int width, int height) {
        List<ImageShape> shapes = computeBackboneShapes(width, height);

        int [] widths = new int[shapes.size()];
        int [] heights = new int[shapes.size()];

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

    List<ImageShape> computeBackboneShapes(int width, int height) {
        List<ImageShape> shapes = new LinkedList<>();

        int[] backboneStrides = maskRCNNConfig.getBackboneStrides();
        for (int i = 0; i < backboneStrides.length; i++) {
            int backboneStride = backboneStrides[i];
            shapes.add(new ImageShape(width/backboneStride, height/backboneStride));
        }
        return shapes;
    }


}
