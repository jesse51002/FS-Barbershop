import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:get_storage/get_storage.dart';

class AppColors {
  static const Color primaryColor = Color(0xFF7A07CE);
  static const Color whiteColor = Color(0xFFFFFFFF);
  static const Color kf0f0f0fColor = Color(0xFFF0F0F0);
  static const Color k030303Color = Color(0xFF030303);
  static const Color k505050Color = Color(0xFF505050);
  static const Color kDEDEDEColor = Color(0xFFDEDEDE);
  static const Color kC7C7F1Color = Color(0xFFC7C7F1);
  static const Color kC2C2C2Color = Color(0xFFC2C2C2);
  static const Color ke5e5e5Color = Color(0xFFE5E5E5);
}

printData(data) {
  if (kDebugMode) {
    print(data);
  }
}

final getStorage = GetStorage();
