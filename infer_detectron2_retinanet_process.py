from infer_detectron2_retinanet import update_path
from ikomia import utils, core, dataprocess
import copy
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import random
import numpy as np


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class RetinanetParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.cuda = True
        self.conf_thresh = 0.8

    def set_values(self, param_map):
        self.cuda = utils.strtobool(param_map["cuda"])
        self.conf_thresh = float(param_map["conf_thresh"])

    def get_values(self):
        param_map = {
            "cuda": str(self.cuda),
            "conf_thresh": str(self.conf_thresh)
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class Retinanet(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)

        # Create parameters class
        if param is None:
            self.set_param_object(RetinanetParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        # get and set config model
        self.LINK_MODEL = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"
        self.threshold = 0.5
        self.cfg = get_cfg()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        # load config from file(.yaml)
        self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL))
        # download the model (.pkl)
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL)
        self.loaded = False
        self.deviceFrom = ""
        self.predictor = None

    def get_progress_steps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def run(self):
        self.begin_task_run()

        # Temporary fix to clean detection outputs
        self.get_output(1).clear_data()

        # we use seed to keep the same color for our masks + boxes + labels (same random each time)
        random.seed(10)

        # Get input :
        img_input = self.get_input(0)
        src_image = img_input.get_image()

        # Get output :
        output_image = self.get_output(0)

        # Get parameters :
        param = self.get_param_object()

        # Set cache dir in the algorithm folder to simplify deployment
        os.environ["FVCORE_CACHE"] = os.path.join(os.path.dirname(__file__), "models")

        # predictor
        if not self.loaded:
            print("Chargement du modèle")
            if not param.cuda:
                self.cfg.MODEL.DEVICE = "cpu"
                self.deviceFrom = "cpu"
            else:
                self.deviceFrom = "gpu"
            self.loaded = True
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA check and load without CUDA
        elif self.deviceFrom == "cpu" and param.cuda:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            # load config from file(.yaml)
            self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL))
            # download the model (.pkl)
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL)
            self.deviceFrom = "gpu"
            self.predictor = DefaultPredictor(self.cfg)
        # reload model if CUDA not check and load with CUDA
        elif self.deviceFrom == "gpu" and not param.cuda:
            print("Chargement du modèle")
            self.cfg = get_cfg()
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            self.cfg.MODEL.DEVICE = "cpu"
            # load config from file(.yaml)
            self.cfg.merge_from_file(model_zoo.get_config_file(self.LINK_MODEL))
            # download the model (.pkl)
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.LINK_MODEL)
            self.deviceFrom = "cpu"
            self.predictor = DefaultPredictor(self.cfg)

        outputs = self.predictor(src_image)
        class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes")

        self.set_names(class_names)

        # get outputs instances
        output_image.set_image(src_image)
        boxes = outputs["instances"].pred_boxes
        scores = outputs["instances"].scores
        classes = outputs["instances"].pred_classes
        self.emit_step_progress()

        # create random color for masks + boxes + labels
        np.random.seed(10)
        colors = [[0, 0, 0]]
        for i in range(len(class_names)):
            colors.append([random.randint(0, 255),
                           random.randint(0, 255),
                           random.randint(0, 255),
                           255])

        # Show boxes + labels + data
        index = 0
        for box, score, cls in zip(boxes, scores, classes):
            if score > param.conf_thresh:
                x1, y1, x2, y2 = box.cpu().numpy()
                w = float(x2 - x1)
                h = float(y2 - y1)
                cls = int(cls.cpu().numpy())
                self.add_object(index, cls, float(score),
                                            float(x1), float(y1), w, h)
            index += 1

        self.forward_input_image(0, 0)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class RetinanetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_detectron2_retinanet"
        self.info.short_description = "RetinaNet inference model of Detectron2 for object detection."
        self.info.authors = "Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár"
        self.info.article = "Focal Loss for Dense Object Detection"
        self.info.journal = "IEEE International Conference on Computer Vision (ICCV)"
        self.info.year = 2017
        self.info.license = "Apache-2.0 License"
        self.info.documentation_link = "https://detectron2.readthedocs.io/index.html"
        self.info.repository = "https://github.com/Ikomia-hub/infer_detectron2_retinanet"
        self.info.original_repository = "https://github.com/facebookresearch/detectron2"
        self.info.path = "Plugins/Python/Detection"
        self.info.icon_path = "icons/detectron2.png"
        self.info.version = "1.3.0"
        self.info.keywords = "object,facebook,detectron2,detection"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OBJECT_DETECTION"

    def create(self, param=None):
        # Create process object
        return Retinanet(self.info.name, param)
