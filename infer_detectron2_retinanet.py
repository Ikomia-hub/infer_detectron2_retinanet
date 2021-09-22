from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        from infer_detectron2_retinanet.infer_detectron2_retinanet_process import RetinanetFactory
        # Instantiate process object
        return RetinanetFactory()

    def getWidgetFactory(self):
        from infer_detectron2_retinanet.infer_detectron2_retinanet_widget import RetinanetWidgetFactory
        # Instantiate associated widget object
        return RetinanetWidgetFactory()
