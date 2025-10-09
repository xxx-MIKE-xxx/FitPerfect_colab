Pipeline „gotowy do pracy”


0) Preflight 1.5 s (YOLO → RTMPose na ORT)

Cel: Zmierzyć realny koszt Twojego 2D-pipeline (detekcja „person” YOLO + top-down RTMPose) na tym urządzeniu i zdecydować: live czy post-process. Wykonanie przez ONNX Runtime (Android/iOS: NNAPI/Core ML dostępne) — w Flutterze użyj flutter_onnxruntime lub onnxruntime z Pub. 
onnxruntime.ai
+2
Dart packages
+2

Wejście testu: 1 klatka z kamery (ostatnia dostępna) lub syntetyczny tensor; rozmiar wejścia YOLO: 640×640 (letterbox), RTM zgodnie z wybranym wariantem (np. 256×192).
Delegaty: preferencja: Core ML (iOS) / NNAPI (Android) → CPU. Częściowe fallbacki operatorów są normalne. 
GitHub

Procedura (1.5 s):

Załaduj sesje ORT dla YOLO i RTMPose.

1× warmup całego łańcucha (YOLO → wybór największego bbox → crop/letterbox → RTMPose).

W pętli przez ~1.5 s wykonuj pełen łańcuch i zapisuj czasy t_chain_ms (YOLO + pre+post + RTM).

Policz medianę i p90 z próbek.

Decyzja trybu + częstotliwość próbkowania:

fps_chain_max = 1000 / median(t_chain_ms)

fps_2d_target = floor(0.8 * fps_chain_max) (20% bufor)

Tryb LIVE, jeśli fps_2d_target ≥ 8. W przeciwnym razie POST.

Ustal stały target sampling czasu: Δt = 1000 / fps_2d_target ms.

Dla kamery 30 FPS wyznacz N = ceil(30 / fps_2d_target) — tylko by orientacyjnie logować, realnie sampluj po timestampach co ~Δt.

Uwaga: RTMPose to top-down (detektor osoby + jeden człowiek), COCO-17; pasuje do naszego łańcucha. 
DeepWiki
+1

1) Nagrywanie (zawsze zapisuj wideo do pliku)

Zawsze rozpocznij zapis MP4 do katalogu sesji (na wypadek późniejszego dokończenia 2D offline).

W LIVE: równolegle uruchom sampler po timestampach (co ~Δt) i odpalaj pipeline 2D dla wybranych klatek z zasadą „latest-frame only” (bufor 1 — brak narastania laga).

Jeśli runtime telemetry pokaże wzrost opóźnień (p90 > 1.5× preflight; backlog > 0.5–1 s): zwiększ Δt (obniż fps_2d_target), a gdy dalej źle — dokończ 2D po STOP z pliku.

2) 2D pipeline (YOLO → RTMPose) i zapis wyników
2.1. Wybór osoby i crop

YOLO: wybierz największy bbox klasy „person” na klatce (obsługujemy „główną osobę”).

Zapamiętaj bbox w układzie surowych pikseli (x1, y1, x2, y2), a do pliku zapisz też bbox znormalizowany [x1/W, y1/H, x2/W, y2/H].

Przygotuj wejście RTMPose: letterbox do rozmiaru modelu (np. 256×192), zapisz parametry r, dw, dh, in_w, in_h, out=[W_in, H_in] do rekordu — umożliwia to dokładny un-letterbox wyników do przestrzeni surowej klatki. (Letterbox/ratio-pad to standard w YOLO; odwrotne mapowanie wymaga znajomości skali i padów). 
docs.opencv.org
+2
GitHub
+2

2.2. Inferencja RTMPose i re-projekcja

Uruchom RTMPose na wycięciu.

Odwzoruj 17 punktów COCO z przestrzeni wejściowej modelu z powrotem do surowej klatki (piksele), odwracając letterbox (skala r, pady dw, dh).

Każdy keypoint zapisujemy jako [x_px, y_px, conf] (piksele raw frame po un-letterbox, nie znormalizowane).

2.3. Format rekordu (JSONL w coco_2d.jsonl)

Jedna linia = jedna przetworzona klatka.

{
  "t": 3.200,                  // sekundy od startu nagrania (float)
  "ts_ms": 3200,               // monotoniczny timestamp w ms
  "frame_idx": 96,             // indeks klatki ze strumienia kamery
  "frame_size": [720,1280],    // [W,H] surowej klatki
  "bbox_norm": [0.22,0.12,0.55,0.88],   // [x1/W, y1/H, x2/W, y2/H]
  "lb_params": {"r":0.56,"dw":40,"dh":12,"in_w":720,"in_h":1280,"out":[640,640]},
  "kpt_coco": [[x_px, y_px, conf], ... 17 szt. ...]
}


Stałe i założenia:

2D keypoints: piksele w przestrzeni surowej klatki (po un-letterbox).

BBox: znormalizowany do [0,1] (względem surowej klatki).

Letterbox info: pełny zestaw, by każdy mógł odtworzyć rzutowania.

2.4. Pliki sesji i lokalizacja

iOS: <App Documents>/FitPerfect/<sessionId>/
np. /var/mobile/Containers/Data/Application/<UUID>/Documents/FitPerfect/<sessionId>/

Android: <App files>/FitPerfect/<sessionId>/
np. /data/user/0/<pkg>/files/FitPerfect/<sessionId>/
(baza ze getApplicationDocumentsDirectory())

Wewnątrz folderu sesji:

video.mp4 (tymczasowy, chyba że zostawiasz go dla użytkownika)

coco_2d.jsonl (główny strumień 2D)

[opcjonalnie debug] yolo_decode.txt, quick_stats.json

3) Po STOP — domknięcie 2D i wejście w 3D (MotionBERT)

Po naciśnięciu STOP:

LIVE: policz tylko brakujące zakresy (wg ts_ms) z pliku wideo i scal z wynikami live → posortuj po ts_ms.

POST: policz pełne 2D z pliku przy tym samym Δt.

Przygotowanie do 3D (robione w motionbert_runner.dart):

Resampling do stałego kroku (np. 8 FPS) wg ts_ms.

Normalizacja do ~[-1,1]: środek w miednicy (pelvis), skala min(W,H)/2.

Okna czasowe: MotionBERT obsługuje długości ≤243 klatek; użyj np. T=243, stride=81. 
Hugging Face
+1

Konwersja COCO-17 → H36M-17 (poniżej).

Wyniki 3D:

out_3d.json — finalna sekwencja [T,17,3] (układ H36M, zcentrowany w pelvis, znormalizowany do ~[-1,1]).

[opcjonalnie debug] mb_input_seq.npy, mb_output_3d.npy, meta.json

MotionBERT oficjalnie używa H36M-17 i okien do 243; repozytorium zawiera skrypty konwersji i inferencji „in-the-wild”. 
GitHub
+1

4) Konwersja COCO-17 → H36M-17 (dla MotionBERT)

COCO indeksy:
0 nose, 1 leye, 2 reye, 3 lear, 4 rear, 5 lsho, 6 rsho, 7 lelb, 8 relb, 9 lwri, 10 rwri, 11 lhip, 12 rhip, 13 lknee, 14 rknee, 15 lank, 16 rank

H36M kolejność (wymagana przez MB):
Pelvis, RHip, RKnee, RAnkle, LHip, LKnee, LAnkle, Spine1, Neck, Head, Site, LShoulder, LElbow, LWrist, RShoulder, RElbow, RWrist 
vision.imar.ro

Mapowanie (piksele w przestrzeni surowej):

Pelvis = mid( LHip(11), RHip(12) )

RHip(12) → RHip

RKnee(14) → RKnee

RAnkle(16) → RAnkle

LHip(11) → LHip

LKnee(13) → LKnee

LAnkle(15) → LAnkle

Spine1 = mid( Pelvis, Neck ) (spina/torso — przybliżenie powszechnie używane w konwersjach do H36M) 
GitHub

Neck = mid( LShoulder(5), RShoulder(6) )

Head = avg( leye(1), reye(2), lear(3), rear(4) ) (głowa powyżej szyi; gdy braki, bierz średnią oczu)

Site = Nose(0) („Site” w H36M odpowiada zwykle punktowi twarzy — praktycznie używa się nosa) 
vision.imar.ro

LShoulder(5) → LShoulder

LElbow(7) → LElbow

LWrist(9) → LWrist

RShoulder(6) → RShoulder

RElbow(8) → RElbow

RWrist(10) → RWrist

Uwaga: Oficjalny kod MotionBERT udostępnia konwersję „coco→h36m” (zob. dataset_wild.py / dataset_action.py), którą można potraktować jako referencję implementacyjną. 
GitHub
+1

5) Śledzenie osoby („główna osoba”)

Na każdej klatce wybieraj największy bbox klasy „person”.

(Opcjonalnie) stabilizuj ID przez IoU matching z poprzednią klatką, ale w razie wielu osób i tak trzymaj tor o największym polu.

6) Metadane i podsumowanie sesji

meta.json (w folderze sesji), np.:

{
  "session_id": "2025-10-09T12-00-01Z_x1y2z3",
  "device": {"platform":"iOS","ep":"coreml","ram_gb":6},
  "video": {"fps":30, "size":[720,1280], "duration_s":12.4},
  "preflight": {"median_ms":118, "p90_ms":140, "fps_target":8, "mode":"live"},
  "sampling": {"dt_ms":125, "strategy":"timestamp"},
  "models": {
    "yolo":{"name":"yolov8n-person.onnx","input":[640,640]},
    "rtmpose":{"name":"rtmpose-s.onnx","input":[256,192]},
    "motionbert":{"name":"MB_ft_h36m.bin","clip_len_max":243}
  },
  "counts": {"frames_2d":98,"clips_3d":3}
}


Po zakończeniu pokaż w UI krótkie podsumowanie (czas trwania, liczba ramek 2D/3D) i przyciski: Otwórz wyniki / Udostępnij folder.

7) Reżimy pracy i degradacja

LIVE: sampler co Δt, bufor 1 (zawsze najnowsza klatka), telemetria (median/p90, backlog).

Fallback: jeśli p90 wyraźnie rośnie lub backlog > 0.5–1 s → zwiększ Δt; jeśli fps_2d_target spadnie < 6 → przejdź w POST (domknięcie 2D po STOP).

POST: całość 2D z pliku z tym samym Δt co w LIVE (spójne wejście dla 3D).

8) Minimalne interfejsy (pseudokod)

Preflight (Dart, ORT; 1.5 s):

final tStart = nowMs();
final samples = <int>[];
do {
  final t0 = nowMs();
  final det = yolo.run(letterbox(frame));                 // bboxy
  final main = pickLargestPerson(det);
  final crop = cropAndLetterbox(frame, main);             // zapis r,dw,dh,in_w,in_h
  final kps = rtmPose.run(crop);                          // 17x3
  final t1 = nowMs();
  samples.add(t1 - t0);
} while (nowMs() - tStart < 1500);

samples.sort();
final median = samples[samples.length ~/ 2];
final fpsTarget = (0.8 * (1000.0 / median)).floor();
final mode = fpsTarget >= 8 ? "live" : "post";
final dtMs = (1000 / max(1, fpsTarget)).round();


Rekord JSONL (LIVE/POST) — patrz sekcja 2.3.

MotionBERT (w motionbert_runner.dart):

wczytaj coco_2d.jsonl → resample do stałego Δt → coco→h36m() → normalizacja (pelvis-center, skala min(W,H)/2) → pakuj okna T=243, stride=81 → infer → out_3d.json.
(MB używa 17 H36M i wspiera długości ≤243). 
Hugging Face

9) Co jeszcze warto mieć (krótkie rekomendacje)

Konfidencja: dla conf < 0.3 — oznacz punkt jako brak; krótko interpoluj (≤5 klatek) przed MB.

Smoothing: lekki EMA/Savitzky–Golay na 2D przed 3D (stabilniejszy MB).

Spójność FPS: zawsze ta sama częstotliwość Δt dla LIVE i POST.

Zależności: ORT Mobile (Android/iOS) i plugin Flutter — aktualne i wspierają mobilne EP. 
onnxruntime.ai
+2
Dart packages
+2

TL;DR — twarde progi i parametry „na start”

Preflight 1.5 s: mierz cały łańcuch (YOLO+RTM).

Decyzja LIVE/POST: fps_2d_target = floor(0.8 * 1000/median_ms); LIVE jeśli ≥ 8 FPS.

Sampling: stały czasowy — Δt = 1000/fps_2d_target ms (np. 125 ms).

Tracking osoby: największy bbox „person” (IoU-stabilizacja opcjonalnie).

Pliki: coco_2d.jsonl (piksele po un-letterbox), out_3d.json (H36M-17), meta.json.

MotionBERT: wejście H36M-17, okna ≤243 (np. 243/81), pelvis-center, skala min(W,H)/2. 
Hugging Face
