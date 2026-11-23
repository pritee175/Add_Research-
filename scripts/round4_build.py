"""Round 4 ingest, stage 2: write the staged Round 4 registers into ad_dataset/round4/ (same schema as ad_dataset/annotations/).
Nothing in ad_dataset/annotations/ is touched, so the frozen v1 splits, features and results stay reproducible.
Labels here are AI labels (ai_ann2-style: transcript + on-screen text + frames watched by the model) and are PENDING HUMAN REVIEW.
Speech comes from the Sarvam cache (.scratch/asr_ocr_cache/*.asr_sarvam.json); on-screen text was read by the model from contact sheets
(first and last sheets of each ad for the long ones, so it is PARTIAL, see README).  Usage: python scripts/round4_build.py"""
import csv, json, os
ROOT = 'E:/Research_FF'; OUT = ROOT + '/ad_dataset/round4'; os.makedirs(OUT, exist_ok=True)
INFO = json.load(open(ROOT + '/.tmp/round4_info.json'))
ADS_COLS = list(csv.DictReader(open(ROOT + '/ad_dataset/annotations/ads.csv', encoding='utf-8')).__next__().keys())
NOTE = 'Third-party YouTube upload ({ch}); round 4 candidate ({tg}), screened as keep on 2026-11-22; metadata AI-filled and unverified.'
# id: (youtube channel, target, title, source_type, product, brand, shown, implied, label, promo_type, surrogate_basis, warning, celeb, lang, year, campaign, conf, reason)
A = {
 'B_0047': ('Innocean India (Hyundai)', 'preventive_alcohol_commercial', 'Hyundai | #BeTheBetterGuy | Road Safety feat. Shah Rukh Khan', 'brand', 'alcohol', 'not_applicable', 'drunk_driving', 'not_applicable', 'preventive', 'not_applicable', '', '', 'yes', 'Hindi-English', '2017', 'CAMP_HYUNDAI_BTBG', 'high',
            'Car-brand commercial-style film (sponsor Hyundai, SRK). Comic ABS-brakes banter ends with "Don\'t drink and drive. Be the better guy." Message is against drink-driving, not for the car.'),
 'B_0048': ('Maruti Suzuki', 'preventive_alcohol_commercial', 'Do not drink and drive', 'brand', 'alcohol', 'not_applicable', 'drunk_driving', 'not_applicable', 'preventive', 'not_applicable', '', '', 'no', 'Hindi-English', '2016', 'CAMP_MARUTI_DD', 'high',
            'Car-brand public-interest spot: drunk men crash at night; on-screen "please do not drive after drinking"; "Issued in public interest by Maruti Suzuki".'),
 'B_0049': ('Black & White Advertising Agency', 'preventive_alcohol_commercial', 'PSA on Drunken Driving', 'other', 'alcohol', 'not_applicable', 'drunk_driving', 'not_applicable', 'preventive', 'not_applicable', '', '', 'no', 'English', '2016', '', 'high',
            'Agency-made PSA with an ironic voice-over that sounds pro-drinking ("Go on, drink") and then shows the crash and the family loss; ends "#Someoneiswaiting". Twist is preventive.'),
 'B_0050': ('(Cyberabad Traffic Police upload)', 'preventive_alcohol_commercial', "A friend's request | Cyberabad Traffic Police | Drunk Driving", 'government', 'alcohol', 'not_applicable', 'drunk_driving', 'not_applicable', 'preventive', 'not_applicable', '', '', 'no', 'Telugu', '2021', '', 'high',
            'Police-issued testimony film in Telugu: a man describes a friend killed by a drunk driver; closes with "Issued in public interest by Cyberabad Traffic Police".'),
 'B_0051': ('Adimoolam L (Dinamalar spot)', 'preventive_alcohol_commercial', 'dinamalar traffic awareness: Drink and Drive', 'other', 'alcohol', 'not_applicable', 'drunk_driving', 'not_applicable', 'preventive', 'not_applicable', '', '', 'no', 'Tamil', '2009', '', 'medium',
            'Tamil newspaper awareness spot, no speech: a hand pours a drink into a car fuel tank, text "alcohol" then "the journey of alcohol is the gateway to death". Label rests on on-screen Tamil text and visuals.'),
 'B_0052': ('NOTORIOUS OG (Chennai Traffic Police film)', 'preventive_alcohol_commercial', 'Chennai Traffic Police Advises (Do Not Drink and Drive)', 'government', 'alcohol', 'not_applicable', 'drunk_driving', 'not_applicable', 'preventive', 'not_applicable', '', '', 'yes', 'Tamil', '2011', '', 'medium',
            'Police drink-driving checkpoint scene (breath analyser, crash) followed by an actor addressing the camera; public message from the traffic police. Samsung boards in the background are incidental. Only partly watched.'),
 'A_0074': ('PaletteAdFilms', 'promotional_tobacco', 'FOUR SQUARE Batting for India', 'other', 'tobacco', 'Four Square', 'cricket_gear', 'cigarettes', 'promotional', 'surrogate_verified', 'Four Square name and four-square logo shared with the Four Square cigarette brand; ad shows cricket gear only, no tobacco.', 'no', 'unknown', 'unknown', 'unknown', 'CAMP_FOURSQUARE', 'high',
            'No speech. Glowing cricket gear and batsman, ends on Four Square logo and "Batting for India": brand-building through cricket, a surrogate for the cigarette brand.'),
 'A_0075': ('90s time travels', 'promotional_tobacco', 'FOUR SQUARE - TEAM INDIA | FOUR SQUARE OLD DOORDARSHAN ADS', 'other', 'tobacco', 'Four Square', 'cricket_gear', 'cigarettes', 'promotional', 'surrogate_verified', 'Four Square name and four-square logo shared with the Four Square cigarette brand; ad shows cricket gear only.', 'no', 'unknown', 'English', 'unknown', 'CAMP_FOURSQUARE', 'high',
            'Old Doordarshan-era spot: Kerala boat race and cricketers, voice-over "Four Square Cricket Gear. You never know what you can become." Surrogate brand promotion.'),
 'A_0076': ('GenesisTVCs', 'promotional_tobacco', 'Wills sport', 'other', 'tobacco', 'Wills', 'sports_apparel', 'cigarettes', 'promotional', 'surrogate_verified', 'Wills name and logo shared with the Wills cigarette brand; ad shows sports apparel and the Wills Indian team.', 'no', 'yes', 'Hindi-English', 'unknown', 'CAMP_WILLS_TEAM', 'high',
            'Cricket-fan montage with the Wills logo, "Wills Sport proudly supports the Wills Indian Team"; surrogate promotion of the Wills name.'),
 'A_0077': ('90s time travels', 'promotional_tobacco', 'CHARMS old ad | Charms Old Doordarshan ads', 'other', 'tobacco', 'Charms', 'cigarette_pack_and_smoking', 'not_applicable', 'promotional', 'direct', '', 'no', 'unknown', 'English', 'unknown', '', 'high',
            'Pre-ban direct cigarette ad: couple by a lake, man lights and smokes a Charms cigarette, pack shown; "Charm is the spirit of freedom". No statutory warning seen in sampled frames.'),
 'A_0078': ('(wills team upload)', 'promotional_tobacco', 'wills team', 'other', 'tobacco', 'Wills', 'sports_apparel', 'cigarettes', 'promotional', 'surrogate_verified', 'Wills name and logo shared with the Wills cigarette brand; ad shows the Indian cricket team and fans.', 'no', 'yes', 'Hindi-English', 'unknown', 'CAMP_WILLS_TEAM', 'high',
            'Fans and team in an "India\'s back to the wall" cricket story, "Come on India!" chant, ends "Wills Sport proudly supports the Wills Indian Team". Surrogate.'),
 'A_0079': ('(Wills Kings upload)', 'promotional_tobacco', 'Wills Kings Commercial (Ca. 1999)', 'other', 'tobacco', 'Wills', 'cigarette_pack_and_smoking', 'not_applicable', 'promotional', 'direct', '', 'no', 'unknown', 'Hindi-English', '1999', '', 'high',
            'VHS rip of a Wills Kings lifestyle ad: friends celebrate a music award, people smoke, pack shown with "Yours." (Yours, mine, ours). Direct cigarette promotion; no warning seen.'),
 'A_0080': ('(Wills team upload)', 'promotional_tobacco', 'Old wills advertisement with Indian team for 96 world cup', 'other', 'tobacco', 'Wills', 'sports_apparel', 'cigarettes', 'promotional', 'surrogate_verified', 'Wills name and logo on team kit; ad shows cricket and fans, no tobacco.', 'no', 'yes', 'Hindi-English', 'unknown', 'CAMP_WILLS_TEAM', 'medium',
            'Long (140 s) Wills cricket music video (slate "Wills Music Video 14-4-99"): stadium fans, team walk-out. Extended cut of the Wills team campaign; surrogate.'),
 'A_0082': ('Royal Stag Barrel Select Shorts', 'promotional_psa_or_csr_style', "A Perfect Short Film That'll Change Your Life | Tinnu Anand", 'brand', 'alcohol', 'Royal Stag', 'none', 'whisky', 'promotional', 'surrogate_verified', "Closing card names Seagram's Royal Stag Barrel Select 'Large Short Films' (brand name shared with Royal Stag whisky); film itself shows no product.", 'no', 'unknown', 'Hindi-English', 'unknown', '', 'medium',
            'Branded short film (about 3 min, cinematic drama) with no product; the whisky brand appears only in the closing sponsor card (#MAKEITPERFECT). Promotional through brand association only; label confidence medium (style vs message test case).'),
}
EXCL = {'A_0081': ('krazykeyur', 'promotional_tobacco', 'Old Indian Doordarshan Ad - Cigarette', 'tobacco', 'Salem', 'not Indian: classic US black-and-white Salem cigarette commercial ("Fresh Air", "Air Softens"); title claimed Doordarshan')}
SCREEN = {  # (start, end, text, type, language)
 'B_0047': [(18, 22, "No..it's second hand. Can't you see, I'm coming straight from the showroom", 'caption', 'English'), (24, 26, 'Looks like top model', 'caption', 'English'), (30, 33, "I don't even consider anything below top model. Plus it's got ABS", 'caption', 'English'),
            (36, 38, "ABS, what's that?", 'caption', 'English'), (48, 51, 'Ok. Suppose car is at top speed and suddenly you have to brake then what happens?', 'caption', 'English'), (66, 69, 'My car has ABS, which means full control on brakes', 'caption', 'English'), (105, 111, '#BeTheBetterGuy', 'tagline', 'English')],
 'B_0048': [(32, 38, 'कृपया शराब पीकर गाड़ी न चलाएं', 'tagline', 'Hindi'), (38, 40, 'Issued in public interest by', 'credit', 'English'), (38, 40, 'MARUTI SUZUKI', 'brand', 'English'), (38, 40, '#SafeRoadsWithMaruti', 'tagline', 'English')],
 'B_0049': [(6, 10, 'you have to hold on to the grass,', 'caption', 'English'), (24, 26, 'and after that', 'caption', 'English'), (108, 114, '#Someoneiswaiting', 'tagline', 'English')],
 'B_0050': [(3, 8, "Antony D'cruz", 'caption', 'English'), (75, 77, 'ISSUED IN PUBLIC INTEREST BY CYBERABAD TRAFFIC POLICE', 'credit', 'English'), (75, 77, 'ప్రజాహితార్థం సైబరాబాద్ ట్రాఫిక్ పోలీస్ వారిచే జారీ', 'credit', 'Telugu'), (75, 77, 'CYBERABAD TRAFFIC POLICE', 'brand', 'English')],
 'B_0051': [(26, 28, 'மது', 'caption', 'Tamil'), (28, 31, 'மதுவின் பயணம் மரணத்தின் வாசல்', 'tagline', 'Tamil'), (28, 31, 'தினமலர்', 'brand', 'Tamil'), (28, 31, 'மக்கள் பணியில் தினமலர் உண்மையின் உரைகல்', 'tagline', 'Tamil')],
 'B_0052': [],
 'A_0074': [(6, 12, 'FOUR SQUARE', 'brand', 'English'), (18, 24, 'FOUR SQUARE', 'brand', 'English'), (20, 24, 'Batting for India', 'tagline', 'English')],
 'A_0075': [(57, 59, 'FOUR SQUARE CRICKET GEAR', 'brand', 'English'), (60, 62, 'You never know what you can become !', 'tagline', 'English'), (60, 62, 'For further information write to P.O. Box No. 222, New Delhi - 110 001', 'other', 'English')],
 'A_0076': [(10, 12, 'WILLS', 'brand', 'English'), (26, 28, 'WILLS SPORT', 'brand', 'English'), (28, 30, 'Wills Sport proudly supports the WILLS INDIAN TEAM', 'tagline', 'English'), (30, 31, 'Apparel & Accessories available at select Club Stores', 'other', 'English')],
 'A_0077': [(78, 80, 'CHARMS', 'brand', 'English')],
 'A_0078': [(44, 48, 'WILLS SPORT proudly supports the WILLS INDIAN TEAM', 'tagline', 'English'), (44, 48, 'WILLS SPORT', 'brand', 'English')],
 'A_0079': [(44, 46, 'WILLS KINGS', 'brand', 'English'), (44, 46, 'YOURS.', 'tagline', 'English'), (40, 44, 'JAZBA', 'other', 'English')],
 'A_0080': [(0, 3, 'WILLS MUSIC VIDEO 14-4-99', 'other', 'English')],
 'A_0082': [(159, 162, "YOU'LL NEVER BE THE SAME AGAIN", 'tagline', 'English'), (165, 168, 'SOME FILMS HAVE THE POWER TO CHANGE YOU', 'tagline', 'English'), (177, 184, "PRESENTING SEAGRAM'S ROYAL STAG BARREL SELECT LARGE SHORT FILMS CATEGORY", 'brand', 'English'),
            (177, 184, '19th MUMBAI FILM FESTIVAL (MAMI) Jio Star', 'credit', 'English'), (177, 184, '#MAKEITPERFECT', 'tagline', 'English')],
}
EVID = {'B_0047': ('speech', 'preventive', 'Closing voice-over: "Don\'t drink and drive. Be the better guy."'), 'B_0048': ('text', 'preventive', 'On-screen "please do not drive after drinking"; issued in public interest by Maruti Suzuki'),
        'B_0049': ('speech', 'preventive', 'Ironic "Go on, drink" voice-over followed by a fatal crash and family loss'), 'B_0050': ('speech', 'preventive', 'Testimony about a friend killed by a drunk driver; issued by Cyberabad Traffic Police'),
        'B_0051': ('text', 'preventive', 'Tamil text: the journey of alcohol is the gateway to death; drink poured into fuel tank'), 'B_0052': ('frame', 'preventive', 'Police drink-driving checkpoint and crash, followed by an actor addressing the camera'),
        'A_0074': ('frame', 'promotional', 'Cricket gear and Four Square logo with "Batting for India"; no tobacco shown (surrogate)'), 'A_0075': ('speech', 'promotional', '"Four Square Cricket Gear. You never know what you can become."'),
        'A_0076': ('text', 'promotional', '"Wills Sport proudly supports the Wills Indian Team" (surrogate for Wills cigarettes)'), 'A_0077': ('frame', 'promotional', 'Man lights a cigarette by the lake, Charms pack shown; "Charm is the spirit of freedom"'),
        'A_0078': ('text', 'promotional', '"Wills Sport proudly supports the Wills Indian Team" after a cricket fan story'), 'A_0079': ('frame', 'promotional', 'Wills Kings pack shown with "Yours."; characters smoking at a celebration'),
        'A_0080': ('frame', 'promotional', 'Wills team cricket music video, Wills logo on kit (surrogate)'), 'A_0082': ('text', 'promotional', "Closing card: Seagram's Royal Stag Barrel Select Large Short Films, #MAKEITPERFECT; no product in the film")}
w = lambda name, cols, rows: csv.writer(open(f'{OUT}/{name}', 'w', newline='', encoding='utf-8')).writerows([cols] + rows)
ads, ann, ev, fr, sc, sp = [], [], [], [], [], []
for ad, (ch, tg, title, st, prod, brand, shown, impl, lab, pt, sb, warn, cel, lang, yr, camp, conf, why) in A.items():
    d = INFO[ad]; yt = d['yt']
    r = dict.fromkeys(ADS_COLS, ''); r.update(ad_id=ad, video_file=f'videos/{ad}.mp4', source_url=f'https://youtu.be/{yt}', youtube_video_id=yt, video_title=title, platform='YouTube', source_type=st, date_collected='2026-11-22',
        year=yr, duration_seconds=d['duration'], language_summary=lang, product_category=prod, brand=brand, shown_product=shown, implied_product=impl, main_label=lab, promotion_type=pt, surrogate_basis=sb,
        statutory_warning_present=warn, celebrity_present=cel, responsible_message_present='', label_confidence=conf, final_reason='AI label (ai_ann2-style pass, frames and transcript watched by the model), PENDING HUMAN REVIEW. ' + why,
        duplicate='no', campaign_id=camp, split='round4', license_or_terms_note=NOTE.format(ch=ch, tg=tg), collected_by='round4', search_target='preventive' if ad.startswith('B') else 'promotional', video_status='downloaded')
    ads.append([r[c] for c in ADS_COLS]); ann.append([ad, 'ai_ann2', lab, conf, why, '', '', ''])
    e = EVID[ad]; ev.append([ad, 0, d['duration'], e[0], e[1], e[2]]); fr += [list(f) for f in d['frames']]
    sc += [[ad, s, e2, t, ty, lg, 'automatic'] for s, e2, t, ty, lg in SCREEN[ad]]
    f = f'{ROOT}/.scratch/asr_ocr_cache/{ad}.asr_sarvam.json'; j = json.load(open(f, encoding='utf-8'))
    sp += [[ad, s['start'], s['end'], s['text'], (s.get('language') or j['language'] or '').split('-')[0], 'automatic'] for s in j['speech']]
for ad, (ch, tg, title, prod, brand, why) in EXCL.items():
    d = INFO[ad]; yt = d['yt']; r = dict.fromkeys(ADS_COLS, ''); r.update(ad_id=ad, video_file=f'videos/{ad}.mp4', source_url=f'https://youtu.be/{yt}', youtube_video_id=yt, video_title=title, platform='YouTube', source_type='other',
        date_collected='2026-11-22', duration_seconds=d['duration'], product_category=prod, brand=brand, main_label='excluded', exclude_reason=why, duplicate='no', license_or_terms_note=NOTE.format(ch=ch, tg=tg), collected_by='round4', search_target='promotional', video_status='downloaded')
    ads.append([r[c] for c in ADS_COLS])
w('ads.csv', ADS_COLS, ads); w('annotators.csv', ['ad_id', 'annotator_id', 'chosen_label', 'confidence', 'reason', 'agreed_flag', 'final_label', 'adjudicator_id'], ann)
w('evidence.csv', ['ad_id', 'start_second', 'end_second', 'evidence_type', 'direction', 'description'], ev); w('frames.csv', ['ad_id', 'second', 'frame_file', 'sampling_method'], fr)
w('screen_text.csv', ['ad_id', 'start_second', 'end_second', 'original_text', 'text_type', 'language', 'correction_status'], sc)
w('speech.csv', ['ad_id', 'start_second', 'end_second', 'original_text', 'language', 'transcription_status'], sp)
print(len(ads), 'ads,', len(ann), 'labels,', len(sc), 'screen rows,', len(sp), 'speech rows,', len(fr), 'frames ->', OUT)
