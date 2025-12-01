def getdistance(widget_list):
    """Fast stub that returns mock distances without VGG16 inference."""
    if not widget_list:
        return []
    return [0.1] * len(widget_list)
