from vector_search import vector_search

if __name__ == '__main__':
  # print("Test: load_mrcnn_model:")
  # vector_search.load_mrcnn_model()

  print("Test: load_mrcnnlite_model:")
  model = vector_search.load_mrcnnlite_model()
  # image_paths = []
  # vector_search.generate_features(image_paths, model)
