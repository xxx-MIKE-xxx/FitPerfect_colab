import 'package:flutter/material.dart';

class VideoQualityScreen extends StatefulWidget {
  const VideoQualityScreen({super.key});

  @override
  State<VideoQualityScreen> createState() => _VideoQualityScreenState();
}

class _VideoQualityScreenState extends State<VideoQualityScreen> {
  String quality = '1080p';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Video quality')),
      body: ListView(
        children: [
          RadioListTile(value: '360p',  groupValue: quality, onChanged: (v) => setState(() => quality = v!), title: const Text('360 p')),
          RadioListTile(value: '720p',  groupValue: quality, onChanged: (v) => setState(() => quality = v!), title: const Text('720 p HD')),
          RadioListTile(value: '1080p', groupValue: quality, onChanged: (v) => setState(() => quality = v!), title: const Text('1080 p Full HD')),
          RadioListTile(value: '4k',    groupValue: quality, onChanged: (v) => setState(() => quality = v!), title: const Text('4 K (beta)')),
        ],
      ),
    );
  }
}
