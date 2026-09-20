"""Authored bidirectional translation fidelity cases; no model judge.

Multiple faithful phrasings are deliberately accepted. These measure semantic
translation review, not unconstrained generation or literary style.
"""


def translation_cases(make_case):
    data = [
        (
            "CS-EN",
            "Czech",
            "English",
            [
                (
                    "Dodavatel nemusí zveřejnit výkres, nesmí jej však předat třetí straně bez předchozího písemného souhlasu zákazníka.",
                    [
                        "The supplier must not publish the drawing, but may disclose it to a third party with the customer's prior written consent.",
                        "The supplier is not required to publish the drawing, but must not pass it to a third party without the customer's prior written consent.",
                        "The supplier need not publish the drawing; however, it may not give it to a third party unless the customer has consented in writing beforehand.",
                        "The supplier need not publish the drawing and may pass it to a third party unless the customer has objected in writing.",
                    ],
                    ["B", "C"],
                ),
                (
                    "Záloha ve výši 1,5 milionu Kč je splatná nejpozději 3. dubna 2027; vratná je pouze při zrušení objednávky dodavatelem.",
                    [
                        "The CZK 1.5 million advance payment is due no later than 3 April 2027; it is refundable only if the supplier cancels the order.",
                        "The CZK 1.5 million backup is due by April 3, 2027 and is refundable if either party cancels.",
                        "The CZK 15 million advance is due no later than 4 March 2027 and refundable only on supplier cancellation.",
                        "An advance of CZK 1.5 million must be paid by 3 April 2027 at the latest and can be refunded only upon cancellation of the order by the supplier.",
                    ],
                    ["A", "D"],
                ),
                (
                    "Ne všechny snímače selhaly. Ověření se týká pouze těch, které byly vyměněny po odstávce, nikoli těch vyměněných před ní.",
                    [
                        "No sensors failed. Verification concerns only those replaced after the shutdown, not before it.",
                        "Not all sensors failed. Verification concerns those replaced before or after the shutdown.",
                        "Not every sensor failed. Only sensors replaced after the shutdown are subject to verification, not those replaced before it.",
                        "Every sensor failed to some extent. Verification applies only to sensors replaced after the shutdown.",
                    ],
                    ["C"],
                ),
            ],
        ),
        (
            "EN-CS",
            "English",
            "Czech",
            [
                (
                    "The operator may restart the robot only after both guards are closed; a closed guard alone does not require a restart.",
                    [
                        "Obsluha musí robota restartovat, jakmile se zavře kterýkoli ochranný kryt.",
                        "Obsluha smí robota restartovat až po zavření obou ochranných krytů; samotné zavření krytu restart nevyžaduje.",
                        "Obsluha smí robota restartovat po zavření alespoň jednoho ochranného krytu; samotné zavření krytu restart nevyžaduje.",
                        "Obsluha může robota znovu spustit pouze tehdy, jsou-li oba ochranné kryty zavřené; zavřený kryt sám o sobě neznamená povinnost restartovat.",
                    ],
                    ["B", "D"],
                ),
                (
                    "The outstanding balance is EUR 2,450.75, excluding tax. The credit note reduces it by EUR 150, not to EUR 150.",
                    [
                        "Neuhrazený zůstatek činí 2 450,75 EUR bez daně. Dobropis jej snižuje o 150 EUR, nikoli na 150 EUR.",
                        "Výjimečný zůstatek činí 2 450,75 EUR včetně daně. Úvěrová zpráva jej snižuje o 150 EUR.",
                        "Dlužná částka je 2 450,75 EUR bez daně. Dobropisem se sníží o 150 EUR, ne na 150 EUR.",
                        "Neuhrazený zůstatek činí 2 450,75 EUR bez daně. Dobropis jej snižuje na 150 EUR, nikoli o 150 EUR.",
                    ],
                    ["A", "C"],
                ),
                (
                    "Had the sample been cooled before weighing, the result might have differed; the report does not say that cooling actually occurred.",
                    [
                        "Vzorek byl před vážením ochlazen, a výsledek se proto určitě lišil; zpráva to potvrzuje.",
                        "Kdyby byl vzorek před vážením ochlazen, výsledek by se nemohl lišit; zpráva neuvádí, zda k ochlazení došlo.",
                        "Kdyby se vzorek po vážení ochladil, výsledek mohl být jiný; zpráva skutečné ochlazení nepotvrzuje.",
                        "Kdyby byl vzorek před vážením ochlazen, výsledek se mohl lišit; zpráva neříká, že k ochlazení skutečně došlo.",
                    ],
                    ["D"],
                ),
            ],
        ),
        (
            "DE-EN",
            "German",
            "English",
            [
                (
                    "Die Anlage darf erst wieder anlaufen, wenn der Druck unter 2 bar liegt; bei genau 2 bar bleibt die Freigabe gesperrt.",
                    [
                        "The plant may restart only once the pressure is below 2 bar; at exactly 2 bar, authorization remains blocked.",
                        "The plant must restart whenever pressure is at most 2 bar; at exactly 2 bar it is enabled.",
                        "The plant may restart once pressure is above 2 bar; at exactly 2 bar it remains blocked.",
                        "The installation may resume operation only when pressure is less than 2 bar; permission is still withheld at exactly 2 bar.",
                    ],
                    ["A", "D"],
                ),
                (
                    "Die Geschäftsführung hat den Vorschlag vorläufig gebilligt, nicht endgültig genehmigt. Eine Umsetzung ist damit noch nicht erlaubt.",
                    [
                        "Management has finally approved the proposal; implementation is now permitted.",
                        "Management has provisionally endorsed the proposal, not given final approval. Implementation is not yet authorized by that endorsement.",
                        "Management has provisionally endorsed the proposal, and this alone authorizes implementation until final approval.",
                        "Management rejected the proposal temporarily; implementation is therefore permanently forbidden.",
                    ],
                    ["B"],
                ),
                (
                    "Die Rechnung ist bis einschließlich 5. Juni zu begleichen. Skonto von 2 % gilt nur für den Nettowarenwert, nicht für Frachtkosten.",
                    [
                        "The invoice must be settled before 5 June, excluding that day. The 2% discount includes freight.",
                        "The invoice is payable by 6 May inclusive. A 2% discount applies only to freight costs.",
                        "The invoice must be paid by 5 June inclusive. A 2% prompt-payment discount applies only to the net value of the goods, not freight charges.",
                        "Payment of the invoice is due no later than 5 June, including that date. The 2% cash discount covers the net goods value alone and excludes freight.",
                    ],
                    ["C", "D"],
                ),
            ],
        ),
        (
            "EN-DE",
            "English",
            "German",
            [
                (
                    "The controller shall retain the last valid reading unless both sensors fail. Failure of one sensor alone must not erase that reading.",
                    [
                        "Die Steuerung muss den letzten gültigen Messwert behalten, es sei denn, beide Sensoren fallen aus. Der Ausfall nur eines Sensors darf diesen Messwert nicht löschen.",
                        "Die Steuerung darf den letzten gültigen Messwert behalten, wenn beide Sensoren ausfallen. Schon ein Ausfall muss ihn löschen.",
                        "Die Steuerung muss den letzten gültigen Messwert behalten, außer wenn beide Sensoren ausfallen. Ein einzelner Sensorausfall darf nicht zum Löschen dieses Werts führen.",
                        "Die Steuerung muss den letzten gültigen Messwert löschen, sobald mindestens ein Sensor ausfällt.",
                    ],
                    ["A", "C"],
                ),
                (
                    "Actual production was 1,200 units, 20% below the forecast, not 20% of it. The revised forecast applies from July onward only.",
                    [
                        "Die aktuelle Produktion betrug 1.200 Einheiten, also 20 % der Prognose. Die neue Prognose gilt rückwirkend ab Juni.",
                        "Die tatsächliche Produktion lag bei 1.200 Einheiten, 20 % unter der Prognose und nicht bei 20 % davon. Die revidierte Prognose gilt erst ab Juli.",
                        "Die tatsächliche Produktion lag bei 1.200 Einheiten, 20 % über der Prognose. Die neue Prognose gilt nur im Juli.",
                        "Tatsächlich wurden 1.200 Einheiten produziert, also 20 % weniger als prognostiziert, nicht 20 % der Prognose. Die überarbeitete Prognose gilt ausschließlich für die Zeit ab Juli.",
                    ],
                    ["B", "D"],
                ),
                (
                    "We cannot rule out a leak, but no leak has been confirmed. Do not describe the test as proving the system leak-free.",
                    [
                        "Wir können ein Leck ausschließen, obwohl es bestätigt wurde. Der Test beweist die Dichtheit.",
                        "Wir haben ein Leck bestätigt, aber es lässt sich nicht lokalisieren. Der Test beweist keine Dichtheit.",
                        "Ein Leck lässt sich nicht ausschließen, doch bislang wurde keines bestätigt. Stellen Sie den Test nicht als Beweis dafür dar, dass das System leckfrei ist.",
                        "Ein Leck ist unmöglich, aber noch nicht bestätigt. Beschreiben Sie den Test als Dichtheitsnachweis.",
                    ],
                    ["C"],
                ),
            ],
        ),
        (
            "CS-DE",
            "Czech",
            "German",
            [
                (
                    "Případné náklady na opravu hradí dodavatel, ledaže závadu způsobilo nesprávné použití; běžné opotřebení se za závadu nepovažuje.",
                    [
                        "Eventuelle Reparaturkosten trägt der Lieferant, es sei denn, der Mangel wurde durch unsachgemäße Verwendung verursacht; normale Abnutzung gilt nicht als Mangel.",
                        "Die endgültigen Reparaturkosten trägt immer der Kunde; normale Abnutzung gilt als Mangel.",
                        "Etwaige Kosten einer Reparatur übernimmt der Lieferant, außer wenn unsachgemäßer Gebrauch den Mangel verursacht hat; gewöhnlicher Verschleiß zählt nicht als Mangel.",
                        "Eventuelle Reparaturkosten trägt der Lieferant nur dann, wenn der Mangel durch unsachgemäße Verwendung verursacht wurde.",
                    ],
                    ["A", "C"],
                ),
                (
                    "Lhůta činí deset pracovních dnů ode dne následujícího po doručení, nikoli deset kalendářních dnů od odeslání.",
                    [
                        "Die Frist beträgt zehn Kalendertage ab Versand, nicht zehn Arbeitstage ab Zustellung.",
                        "Die Frist beträgt zehn Arbeitstage ab dem auf die Zustellung folgenden Tag, nicht zehn Kalendertage ab Versand.",
                        "Die Frist beträgt zehn Arbeitstage einschließlich des Zustellungstags und nicht zehn Kalendertage ab Versand.",
                        "Die Frist beträgt zehn Arbeitstage ab dem Tag nach der Zustellung, nicht zehn Kalendertage ab dem Versand.",
                    ],
                    ["B", "D"],
                ),
                (
                    "Ačkoli nebylo prokázáno, že úprava zvyšuje spotřebu, nelze z toho vyvodit, že ji snižuje.",
                    [
                        "Obwohl nachgewiesen wurde, dass die Änderung den Verbrauch erhöht, muss sie ihn senken.",
                        "Weil die Änderung den Verbrauch nachweislich nicht erhöht, senkt sie ihn zwangsläufig.",
                        "Obwohl nicht nachgewiesen wurde, dass die Änderung den Verbrauch erhöht, lässt sich daraus nicht schließen, dass sie ihn senkt.",
                        "Da die Änderung den Verbrauch senkt, kann eine Erhöhung ausgeschlossen werden.",
                    ],
                    ["C"],
                ),
            ],
        ),
        (
            "DE-CS",
            "German",
            "Czech",
            [
                (
                    "Der Auftragnehmer muss die Unterlagen nicht veröffentlichen, darf sie aber ohne Zustimmung auch nicht an Dritte weitergeben.",
                    [
                        "Zhotovitel nesmí podklady zveřejnit, ale smí je bez souhlasu předat třetím osobám.",
                        "Zhotovitel nemusí podklady zveřejnit, bez souhlasu je však nesmí ani předat třetím osobám.",
                        "Zhotovitel má povinnost podklady zveřejnit, ale třetím osobám je předat nemusí.",
                        "Zhotovitel není povinen podklady zveřejňovat, avšak bez souhlasu je také nesmí poskytnout třetím stranám.",
                    ],
                    ["B", "D"],
                ),
                (
                    "Die Freigabe erfolgt frühestens am 12. August, sofern bis dahin beide Prüfberichte vorliegen; ein einzelner Bericht genügt nicht.",
                    [
                        "Ke schválení dojde nejdříve 12. srpna, pokud do té doby budou k dispozici obě zkušební zprávy; jedna zpráva nestačí.",
                        "Ke schválení dojde nejpozději 12. srpna, pokud bude k dispozici alespoň jedna zkušební zpráva.",
                        "Schválení proběhne nejdříve 12. srpna za předpokladu, že do té doby budou předloženy oba protokoly o zkoušce; jediný protokol nepostačuje.",
                        "Schválení proběhne 12. srpna bez ohledu na to, zda jsou zkušební zprávy k dispozici.",
                    ],
                    ["A", "C"],
                ),
                (
                    "Der Wirkungsgrad stieg von 80 % auf 84 %, also um vier Prozentpunkte, nicht um vier Prozent relativ zum Ausgangswert.",
                    [
                        "Účinnost stoupla z 80 % na 84 %, tedy relativně o čtyři procenta původní hodnoty.",
                        "Účinnost klesla z 84 % na 80 %, tedy o čtyři procentní body.",
                        "Účinnost stoupla z 80 % na 84 %, tedy o čtyři procentní body, nikoli relativně o čtyři procenta výchozí hodnoty.",
                        "Účinnost stoupla o 84 procentních bodů na 80 %, nikoli o čtyři procenta.",
                    ],
                    ["C"],
                ),
            ],
        ),
    ]
    cases = []
    for direction, source, target, blocks in data:
        prompt = (
            f"Translate/review {source} into {target}. For EACH source passage choose ALL "
            "candidate translations that preserve its entire meaning: modality, negation, "
            "conditions, actors, quantities, and dates. Several phrasings may be faithful; "
            "do not reject synonyms or word order merely for style. Do not silently repair "
            "a candidate. Letters restart for each passage. Return an object with keys "
            '"1","2","3", each an alphabetically sorted array of accepted letters.\n'
        )
        answer = {}
        for i, (original, candidates, accepted) in enumerate(blocks, 1):
            prompt += f"\nPassage {i} ({source}): {original}\n"
            prompt += (
                "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", candidates)) + "\n"
            )
            answer[str(i)] = accepted
        cases.append(
            make_case(
                f"SP1-TR-{direction}",
                "Language Translations",
                prompt,
                answer,
                f"{source} to {target}: all and only meaning-preserving translations, including alternative faithful phrasings.",
                "Original parallel passages; semantic fidelity selection, not free-form fluency",
            )
        )
    return cases
