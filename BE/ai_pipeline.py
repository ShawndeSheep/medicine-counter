def start_pipeline(image):
    print("Running AI pipeline...")
    
    # 1. Segmentation
    segmentation_result = segmentation(image)
    print(segmentation_result)
    # 2. Embedding
    embedding_result = embedding(image)
    print(embedding_result)
    
    # 3. Clustering
    clustering_result = clustering()
    print   (clustering_result)    
    # 4. Classification
    classification_result = classification()
    print(classification_result)
    
    # 5. Output Formatting
    outputFormatting_result = outputFormatting()
    print(outputFormatting_result)
    return outputFormatting_result

def segmentation(image):
    return "Running segmentation..."

def embedding(image):
    return "Running embedding..."

def clustering():
    return "Running clustering..."

def classification():
    return "Running classification..."

def outputFormatting():
    return "Running output formatting..."