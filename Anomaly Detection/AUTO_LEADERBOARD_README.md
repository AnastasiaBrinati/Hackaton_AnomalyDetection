# 🔄 Aggiornamento Automatico Leaderboard - GUIDA RAPIDA

## ❌ PROBLEMA: La Leaderboard Non Si Aggiorna Automaticamente

Se la leaderboard non si aggiorna automaticamente nel markdown, segui questa guida per risolvere il problema.

## ✅ SOLUZIONE RAPIDA

### 1. Setup Sistema Automatico (DA FARE UNA VOLTA)

```bash
# Attiva l'environment hackathon_env
source hackathon_env/bin/activate

# Esegui il setup automatico
python setup_auto_leaderboard.py
```

Questo script configurerà:
- 🔧 **Git Hook** che si attiva automaticamente sui commit
- 👀 **File Watcher** per monitoraggio in tempo reale
- 📝 **Script manuali** per aggiornamenti rapidi

### 2. Test del Sistema

```bash
# Test manuale (sempre funziona)
./update_leaderboard_manual.sh

# Test automatico - crea una submission di prova
cd Track1_Solution
python track1_anomaly_detection.py
cd ..

# Committa la submission - dovrebbe triggerare l'aggiornamento automatico
git add submissions/submission_me_giorgio.json
git commit -m "Test submission - trigger auto leaderboard"
```

## 🎯 Modalità di Utilizzo Durante l'Hackathon

### Opzione 1: Git Hook Automatico (ZERO SFORZO) ⭐
```bash
# Nessuna azione richiesta!
# Il sistema si attiva automaticamente quando i partecipanti fanno commit di submissions
```

### Opzione 2: File Watcher (TEMPO REALE)
```bash
# Avvia in una finestra terminale separata
python file_watcher.py
# Lascia girare durante tutto l'hackathon
# Si aggiorna in tempo reale quando i file cambiano
```

### Opzione 3: Aggiornamento Manuale (BACKUP)
```bash
# Quando vuoi aggiornare manualmente
./update_leaderboard_manual.sh
```

## 🚨 Risoluzione Problemi Comuni

### "Git hook non funziona"
```bash
# Verifica che il hook esista e sia eseguibile
ls -la .git/hooks/post-commit
chmod +x .git/hooks/post-commit

# Re-run setup se necessario
python setup_auto_leaderboard.py
```

### "File watcher crashes"
```bash
# Installa dipendenza mancante
pip install watchdog

# Riavvia
python file_watcher.py
```

### "Environment non trovato"
```bash
# Assicurati di essere nella root directory del progetto
ls hackathon_env/

# Attiva environment
source hackathon_env/bin/activate

# Poi esegui setup
python setup_auto_leaderboard.py
```

### "Python command not found"
```bash
# Se sei nell'environment hackathon_env, usa python
# Se non sei nell'environment, potrebbe servire python3

# Prova:
python3 setup_auto_leaderboard.py
# oppure
./update_leaderboard_manual.sh
```

## 📋 Checklist Veloce

Per gli **organizzatori** prima dell'hackathon:

1. [ ] `source hackathon_env/bin/activate`
2. [ ] `python setup_auto_leaderboard.py`
3. [ ] Verifica: `ls -la .git/hooks/post-commit`
4. [ ] Test: `./update_leaderboard_manual.sh`
5. [ ] ✅ **PRONTO!**

Per i **partecipanti** durante l'hackathon:

1. [ ] Modifica `team_name` e `members` nel codice
2. [ ] Esegui `python track1_anomaly_detection.py`
3. [ ] `git add submissions/submission_[team].json`
4. [ ] `git commit -m "Team submission"`
5. [ ] `git push origin main`
6. [ ] ✅ **Leaderboard si aggiorna automaticamente!**

## 🎉 Risultato Atteso

Dopo il setup corretto:

- ✅ Ogni **commit con submission** trigge aggiornamento automatico
- ✅ La **leaderboard.md** si aggiorna automaticamente
- ✅ Le **modifiche vengono committate** automaticamente
- ✅ **Zero intervento manuale** richiesto durante l'hackathon
- ✅ **Backup manuali** sempre disponibili

## 📞 Help

Se continui ad avere problemi:

1. **Controlla i log** quando esegui i comandi
2. **Verifica permessi** sui file di script
3. **Assicurati di essere nell'environment** hackathon_env
4. **Prova l'aggiornamento manuale** come fallback: `./update_leaderboard_manual.sh`

---

**🚀 Con questo sistema la leaderboard si aggiornerà automaticamente ad ogni submission! 🏆** 