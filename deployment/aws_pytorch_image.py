from sagemaker import image_uris
print(image_uris.retrieve(framework='pytorch', region='us-east-1', version='2.0.0', py_version='py310', image_scope='inference', instance_type='ml.g5.xlarge'))