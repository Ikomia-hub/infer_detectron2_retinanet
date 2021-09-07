from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class Detectron2_RetinaNet(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from Detectron2_RetinaNet.Detectron2_RetinaNet_process import Detectron2_RetinaNetProcessFactory
        # Instantiate process object
        return Detectron2_RetinaNetProcessFactory()

    def getWidgetFactory(self):
        from Detectron2_RetinaNet.Detectron2_RetinaNet_widget import Detectron2_RetinaNetWidgetFactory
        # Instantiate associated widget object
        return Detectron2_RetinaNetWidgetFactory()
