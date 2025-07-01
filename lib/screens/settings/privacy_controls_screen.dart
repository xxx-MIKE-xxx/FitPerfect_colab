import 'package:flutter/material.dart';

class PrivacyControlsScreen extends StatefulWidget {
  const PrivacyControlsScreen({super.key});

  @override
  State<PrivacyControlsScreen> createState() => _PrivacyControlsScreenState();
}

class _PrivacyControlsScreenState extends State<PrivacyControlsScreen> {
  bool shareData = false;
  bool personalizedAds = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Privacy controls')),
      body: ListView(
        children: [
          SwitchListTile(
            value: shareData,
            onChanged: (v) => setState(() => shareData = v),
            title: const Text('Share anonymous usage data'),
          ),
          SwitchListTile(
            value: personalizedAds,
            onChanged: (v) => setState(() => personalizedAds = v),
            title: const Text('Allow personalized ads'),
          ),
        ],
      ),
    );
  }
}
