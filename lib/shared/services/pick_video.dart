import 'dart:io';
import 'package:image_picker/image_picker.dart';

/// Returns a temporary **File** pointing at the selected video,
/// or `null` if the user cancelled.
Future<File?> pickVideoFromGallery() async {
  final xfile = await ImagePicker().pickVideo(
    source: ImageSource.gallery,
    maxDuration: const Duration(minutes: 5),   // optional
  );
  return xfile == null ? null : File(xfile.path);
}
