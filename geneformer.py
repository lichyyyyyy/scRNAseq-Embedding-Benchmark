from models.Geneformer.geneformer import EmbExtractor

embex = EmbExtractor(model_type="CellClassifier",
        num_classes=3,
        emb_mode="cell",
        filter_data={"cell_type":["cardiomyocyte"]},
        max_ncells=1000,
        emb_layer=-1,
        emb_label=["disease", "cell_type"],
        labels_to_plot=["disease", "cell_type"])