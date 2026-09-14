class EndPoints {
  static String baseUrl = "http://192.168.29.236:8000/hairstyle_creation";

  Uri getStarted() {
    return Uri.parse("$baseUrl/start");
  }

  Uri hairStylePreset({required String eventId}) {
    return Uri.parse("$baseUrl/hairstyle_presets?eventid=$eventId");
  }

  Uri qrLink({required String eventId}) {
    return Uri.parse("$baseUrl/custom_img/link?eventid=$eventId");
  }

  Uri getImageId({required String eventId}) {
    return Uri.parse("$baseUrl/custom_img/imageid?eventid=$eventId");
  }

  Uri startRendering({required String eventId}) {
    return Uri.parse("$baseUrl/rendering/start/?eventid=$eventId");
  }

  Uri getRenderingResult({required String eventId}) {
    return Uri.parse("$baseUrl/rendering/results?eventid=$eventId");
  }
}
