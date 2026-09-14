import 'package:get/get.dart';
import 'package:hair_style_app/screens/on_boarding/on_boarding_screen.dart';

class SplashController extends GetxController {
  @override
  void onInit() {
    Future.delayed(
      const Duration(seconds: 3),
          () {
        Get.offAll(const OnBoardingScreen());
      },
    );
    super.onInit();
  }
}
