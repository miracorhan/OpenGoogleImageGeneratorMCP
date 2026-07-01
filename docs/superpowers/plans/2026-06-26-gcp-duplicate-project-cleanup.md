# GCP Duplicate Project Cleanup Plan (Revize)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `idle-defense-496111` projesini doğrulayıp sil; diğer çiftlerin farklı roller üstlendiğini belgeleyerek gelecekteki karışıklığı önle.

**Architecture:** Firebase-suffix projeler gerçek mobil Firebase projeleridir (Auth, FCM, Firestore). GCP projeler backend/publish rolleri üstlenir. İkisi "duplicate" değil, birbirini tamamlar. Tek gerçek orphan: `idle-defense-496111` — Firebase enabled ama hiç uygulama kaydı yok, billing açık.

**Tech Stack:** gcloud CLI, Firebase MCP

## Proje Rolleri (Kesinleşmiş)

| Uygulama | Firebase Projesi (mobil app) | GCP Projesi (backend) | Silinecek mi? |
|----------|------------------------------|-----------------------|---------------|
| EveryTranslate | `everytranslate-b7d30` — com.everytranslate.app | `everytranslate-499108` — app-publish SA | **HİÇBİRİ** |
| InviteQuick | `invitequick-99f6a` — com.ruicodex.invitequick | `invitequick` — GeminiApp, RevenueCat, compute SAs | **HİÇBİRİ** |
| Idle Defense | `idle-defense-89707` — com.ruicodex.idledefense ← gerçek | `idle-defense-496111` — Firebase var ama app YOK | **`idle-defense-496111` silinecek** |
| PosterBoost | `posterboost-ad51c` — com.ruicodex.posterboost.ai | `gen-lang-client-0696014150` — play-publish, RevenueCat SAs | **HİÇBİRİ** |
| RadioSEM | `radiosem-c950d` — com.ruicodex.radiosem | `radiosem` — RevenueCat SA | **HİÇBİRİ** |

## Kök Sebep

Firebase Console'da "Add Firebase to existing GCP project" yerine "Create new project" seçildiğinde Firebase yeni bir proje yaratır. Bu yüzden her uygulamanın iki projesi var: biri GCP backend, biri Firebase mobil. Bu bir hata değil, bir tasarım tercihi — ama yanlış anlaşılınca "duplicate" gibi görünüyor.

**Tek gerçek hata:** `idle-defense-496111` — 15 dakika sonra yaratılmış, Firebase eklenmiş, billing açılmış ama hiç uygulama kaydedilmemiş. Gerçek Firebase projesi `idle-defense-89707`.

---

## Task 1: idle-defense-496111 — Son doğrulama

**Amaç:** Projeyi silmeden önce herhangi bir aktif kaynak, data veya dependency olmadığını teyit et.

- [ ] **Step 1: Auth yenile**

```
gcloud auth login info@ruicodex.com
```

- [ ] **Step 2: Firestore verisi var mı?**

```
gcloud firestore databases list --project=idle-defense-496111 --format=json
```
Beklenen: boş liste `[]` veya sadece `(default)` database, içinde collection yok.

Firebase Console → `idle-defense-496111` → Firestore → Collections sekmesi boş olmalı.

- [ ] **Step 3: Cloud Storage bucket'ı var mı?**

```
gcloud storage buckets list --project=idle-defense-496111 --format="value(name)"
```
Beklenen: boş çıktı veya sadece Firebase auto-created bucket (`idle-defense-496111.appspot.com`) — içi boş olmalı.

- [ ] **Step 4: Firebase Auth kullanıcısı var mı?**

Firebase Console → `idle-defense-496111` → Authentication → Users sekmesi.
Beklenen: 0 kullanıcı.

- [ ] **Step 5: Billing alert veya budget var mı?**

```
gcloud billing budgets list --billing-account=012192-A57B5B-861B86 --format=json
```
`idle-defense-496111` için tanımlı budget varsa önce sil.

- [ ] **Step 6: Sonuçları kaydet**

```
Firestore boş:  EVET / HAYIR
Storage boş:    EVET / HAYIR
Auth kullanıcı: 0 / ???
Budget yok:     EVET / HAYIR
```
Hepsi EVET → Task 2'ye geç.
Herhangi biri HAYIR → önce o kaynağı değerlendir.

---

## Task 2: idle-defense-496111 projesini sil

- [ ] **Step 1: Billing bağlantısını kes**

```
gcloud billing projects unlink idle-defense-496111
```

- [ ] **Step 2: Projeyi sil**

```
gcloud projects delete idle-defense-496111
```
Onay promptunda: `y`

- [ ] **Step 3: Silmeyi doğrula**

```
gcloud projects describe idle-defense-496111
```
Beklenen: `lifecycleState: DELETE_REQUESTED`

- [ ] **Step 4: Mobil uygulamanın hâlâ çalıştığını doğrula**

`idle-defense-89707` projesi etkilenmemeli — bu adımı atlama.
Firebase Console → `idle-defense-89707` → Uygulama hâlâ `com.ruicodex.idledefense` olarak kayıtlı mı? ✅

---

## Task 3: RadioSEM billing boşluğunu değerlendir

**Bağlam:** `radiosem` (GCP backend projesi) ve `radiosem-c950d` (Firebase mobil projesi) — **ikisinde de billing yok.** RevenueCat entegrasyonu veya Vertex AI kullanıyorsa billing açılmalı.

- [ ] **Step 1: radiosem aktif kullanıyor mu kontrol et**

```
gcloud services list --enabled --project=radiosem --format="value(config.name)"
```
`aiplatform.googleapis.com` veya `run.googleapis.com` varsa billing açılmalı.

- [ ] **Step 2: Gerekirse billing ekle**

```
gcloud billing projects link radiosem --billing-account=012192-A57B5B-861B86
```

---

## Task 4: Gelecekte duplicate önleme

**Firebase eklenirken doğru akış:**

```
Firebase Console → Add project →
"Add Firebase to a Google Cloud project" seçeneğini seç →
Mevcut GCP projesini seç (örn. invitequick, gen-lang-client-0696014150)
```

Bu şekilde tek bir projede hem billing/backend hem Firebase özellikleri olur.

**Mevcut çiftler için migration gerekli mi?**
Hayır. Mevcut setup (ayrı Firebase + GCP projeler) çalışıyor. Migration yüksek risk, düşük fayda.

---

## Özet: Sonraki Adımlar

1. `idle-defense-496111` → Task 1 doğrulama → Task 2 silme
2. RadioSEM billing → Task 3
3. Diğer 4 çift: **dokunma**, ayrı roller üstleniyor

**Silmeye aday olan sadece: `idle-defense-496111`**
