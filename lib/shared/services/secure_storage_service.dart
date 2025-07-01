import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class SecureStorageService {
  const SecureStorageService();

  static const _ss = FlutterSecureStorage();

  Future<void> write(String key, String value) =>
      _ss.write(key: key, value: value);

  Future<String?> read(String key) =>
      _ss.read(key: key);

  Future<void> delete(String key) =>
      _ss.delete(key: key);
}
