import update_path
from ikomia import core, dataprocess
import copy
# Your imports below
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import random

# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class Detectron2_RetinaNetParam(core.CProtocolTaskParam):

    def __init__(self):
        core.CProtocolTaskParam.__init__(self)
        self.cuda = True
        self.proba = 0.8

    def setParamMap(self, paramMap):
        self.cuda = int(paramMap["cuda"])
        self.proba = int(paramMap["proba"])

    def getParamMap(self):
        paramMap = core.ParamMap()
        paramMap["cuda"] = str(self.cuda)
        paramMap["proba"] = str(self.proba)
        return paramMap


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class Detectron2_RetinaNetProcess(dataprocess.CImageProcess2d):

    def __init__(self, name, param):
        dataprocess.CImageProcess2d.__init__(self, name)

        #Create parameters class
        if param is None:
            self.setParam(Detectron2_RetinaNetParam())
        else:
            self.setParam(copy.deepcopy(param))

        # get and set config model
        self.LINK_MODEL = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        self.threshold = 0.5
        self.cfg = get_cfg()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL)) # load config from file(.yaml)
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL) # download the model (.pkl)
        self.loaded = False
        self.deviceFrom = ""

        # add output
        self.addOutput(dataprocess.CGraphicsOutput())

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        self.beginTaskRun()

        # we use seed to keep the same color for our masks + boxes + labels (same random each time)
        random.seed(30)

        # Get input :
        input = self.getInput(0)
        srcImage = input.getImage()

        # Get output :
        output_image = self.getOutput(0)
        output_graph = self.getOutput(1)
        output_graph.setNewLayer("Detectron2_RetinaNet")

        # Get parameters :
        param = self.getParam()

        # predictor
        if not self.loaded:
            print("Chargement du modèle")
            if param.cuda == False:
                self.cfg.MODEL.DEVICE = "cpu"
                self.deviceFrom = "cpu"
            else:
                self.deviceFrom = "gpu"
            self.loaded = True
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA check and load without CUDA 
        elif self.deviceFrom == "cpu" and param.cuda == True:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL)) # load config from file(.yaml)
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL) # download the model (.pkl)
            self.deviceFrom = "gpu"
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and param.cuda == False:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.MODEL.DEVICE = "cpu"
            self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL)) # load config from file(.yaml)
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL) # download the model (.pkl)
            self.deviceFrom = "cpu"
            self.predictor = DefaultPredictor(self.cfg)
        
        outputs = self.predictor(srcImage)

        # get outputs instances
        output_image.setImage(srcImage)
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores
        classes = outputs["instances"].pred_classes

        # to numpy
        if param.cuda :
            boxes_np = boxes.tensor.cpu().numpy()
            scores_np = scores.cpu().numpy()
            classes_np = classes.cpu().numpy()
        else :
            boxes_np = boxes.tensor.numpy()
            scores_np = scores.numpy()
            classes_np = classes.numpy()

        self.emitStepProgress()
        
        # keep only the results with proba > threshold
        scores_np_tresh = list()
        for s in scores_np:
            if float(s) > param.proba:
                scores_np_tresh.append(s)

        if len(scores_np_tresh) > 0:
            # text label with score
            labels = None
            class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")
            if classes is not None and class_names is not None and len(class_names) > 1:
                labels = [class_names[i] for i in classes]
            if scores_np_tresh is not None:
                if labels is None:
                    labels = ["{:.0f}%".format(s * 100) for s in scores_np_tresh]
                else:
                    labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores_np_tresh)]

            # Show Boxes + labels 
            for i in range(len(scores_np_tresh)):
                color = [random.randint(0,255), random.randint(0,255), random.randint(0,255), 255]
                properties_text = core.GraphicsTextProperty()
                properties_text.color = color
                properties_text.font_size = 7
                properties_rect = core.GraphicsRectProperty()
                properties_rect.pen_color = color
                output_graph.addRectangle(float(boxes_np[i][0]), float(boxes_np[i][1]), float(boxes_np[i][2] - boxes_np[i][0]), float(boxes_np[i][3] - boxes_np[i][1]), properties_rect)
                output_graph.addText(labels[i],float(boxes_np[i][0]), float(boxes_np[i][1]),properties_text)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class Detectron2_RetinaNetProcessFactory(dataprocess.CProcessFactory):

    def __init__(self):
        dataprocess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "Detectron2_RetinaNet"
        self.info.shortDescription = "RetinaNet inference model of Detectron2 for object detection."
        self.info.description = "RetinaNet inference model for object detection trained on COCO. " \
                                "Implementation from Detectron2 (Facebook Research). " \
                                "This Ikomia plugin can make inference of pre-trained model " \
                                "with ResNet101 backbone + FPN head."
        self.info.authors = "Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár"
        self.info.article = "Focal Loss for Dense Object Detection"
        self.info.journal = "IEEE International Conference on Computer Vision (ICCV)"
        self.info.year = 2017
        self.info.license = "Apache-2.0 License"
        self.info.documentationLink = "https://detectron2.readthedocs.io/index.html"
        self.info.repo = "https://github.com/facebookresearch/detectron2"
        self.info.path = "Plugins/Python/Detectron2"
        self.info.iconPath = "icons/detectron2.png"
        self.info.version = "1.0.1"
        self.info.keywords = "object,facebook,detectron2,detection"

    def create(self, param=None):
        # Create process object
        return Detectron2_RetinaNetProcess(self.info.name, param)
