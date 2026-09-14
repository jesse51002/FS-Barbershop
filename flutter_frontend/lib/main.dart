import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:get_storage/get_storage.dart';
import 'package:hair_style_app/confing/app_colors.dart';
import 'package:hair_style_app/confing/localization.dart';
import 'package:hair_style_app/confing/root_binding.dart';
import 'package:hair_style_app/screens/splash/splash_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await GetStorage.init();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return GetMaterialApp(
      initialBinding: RootBinding(),
      translations: Localization(),
      locale: const Locale('en', 'US'),
      debugShowCheckedModeBanner: false,
      fallbackLocale: const Locale('en', 'US'),
      theme: ThemeData(
          fontFamily: "Montserrat-Regular",
          scrollbarTheme: ScrollbarThemeData().copyWith(
            thumbColor: MaterialStateProperty.all(AppColors.ke5e5e5Color),
          )),
      home: const SplashScreen(),
    );
  }
}
