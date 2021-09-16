def build_model(cfg, mode="train"):
    if cfg.MODEL.CHOICE == 'RotNet':
        from CompNet.models.RotNet import build_RotNet
        net, loss, metric = build_RotNet(cfg, mode)
    elif cfg.MODEL.CHOICE == 'JointNet':
        from CompNet.models.JointNet import build_JointNet
        net, loss, metric = build_JointNet(cfg, mode)
    elif cfg.MODEL.CHOICE == 'AxisLengthNet':
        from CompNet.models.AxisLengthNet import build_AxisLengthNet
        net, loss, metric = build_AxisLengthNet(cfg, mode)
    elif cfg.MODEL.CHOICE == 'GroupAxisLengthNet':
        from CompNet.models.GroupAxisLengthNet import build_GroupAxisLengthNet
        net, loss, metric = build_GroupAxisLengthNet(cfg, mode)
    elif cfg.MODEL.CHOICE == 'SizeRelationNet':
        from CompNet.models.SizeRelationNet import build_SizeRelationNet
        net, loss, metric = build_SizeRelationNet(cfg, mode)
    else:
        raise NotImplementedError()
    print(f'Empty build_model, add build_model')
    return net, loss, metric
