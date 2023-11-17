class PathRecordsConverter:
    def image_size(self, size, CFG):
        self.repo = CFG.GCS_REPO
        default = None
        return getattr(self, f"image_size_{size}", default)()
    
    def image_size_192(self):
        GCS_PATH = f"gs://{self.repo}/tfrecords-jpeg-192x192"
        NUM_RECORDS = 25
        return GCS_PATH, NUM_RECORDS
    
    def image_size_224(self):
        GCS_PATH = f"gs://{self.repo}/tfrecords-jpeg-224x224v2"
        NUM_RECORDS = 33
        return GCS_PATH, NUM_RECORDS
    
    def image_size_256(self):
        GCS_PATH = f"gs://{self.repo}/tfrecords-jpeg-256x256"
        NUM_RECORDS = 50
        return GCS_PATH, NUM_RECORDS
    
    def image_size_384(self):
        GCS_PATH = f"gs://{self.repo}/tfrecords-jpeg-384x384"
        NUM_RECORDS = 75
        return GCS_PATH, NUM_RECORDS
    
    def image_size_512(self):
        GCS_PATH = f"gs://{self.repo}/tfrecords-jpeg-512x512"
        NUM_RECORDS = 100
        return GCS_PATH, NUM_RECORDS
    
    def image_size_(self):
        GCS_PATH = f"gs://{self.repo}/tfrecords-jpeg-raw"
        NUM_RECORDS = 200
        return GCS_PATH, NUM_RECORDS