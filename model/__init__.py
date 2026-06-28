"""模型模块"""
# 延迟导入，避免在未安装PyTorch时导入失败
def __getattr__(name):
    if name in ('ModelEncoderMlpSimple', 'ModelTwoLayer', 'ModelLinear',
                'MODEL_DICT', 'create_model', 'load_model', 'save_model'):
        from model.network import (
            ModelEncoderMlpSimple, ModelTwoLayer, ModelLinear,
            MODEL_DICT, create_model, load_model, save_model
        )
        return locals().get(name)
    elif name == 'encode_game_state':
        from model.nn_input import encode_game_state
        return encode_game_state
    elif name == 'HandwrittenEvaluator':
        from model.handwritten import HandwrittenEvaluator
        return HandwrittenEvaluator
    raise AttributeError(f"module 'model' has no attribute {name}")
