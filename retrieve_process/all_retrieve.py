import sys
from pathlib import Path

# Pastikan bisa import file lain di retrieve_process
sys.path.insert(0, str(Path(__file__).parent))

def run_mpm_orp():
    """
    MPM ORP Retrieval
    Menggunakan pipeline existing: retrieve.py
    """
    print("\n[MODE] MPM ORP selected\n")
    try:
        import retrieve
        retrieve.main()
    except Exception as e:
        print(f"[ERROR] Failed to run MPM ORP retrieval: {e}")


def run_mpm_lm():
    """
    MPM LM Retrieval (STUB)
    TODO:
    - Tentukan collection LM internal
    - Tentukan apakah pakai rerank
    - Tentukan input router / NER
    """
    print("\n[MODE] MPM LM selected\n")
    print("[INFO] MPM LM retrieval is NOT IMPLEMENTED yet.")
    print("[TODO] This pipeline will be added in the future.\n")


def run_confluence_lm():
    """
    Confluence LM Retrieval
    Menggunakan pipeline rerank: retrieve_rerank_confluence.py
    """
    print("\n[MODE] Confluence LM selected\n")
    try:
        import retrieve_rerank_confluence
        retrieve_rerank_confluence.main()
    except Exception as e:
        print(f"[ERROR] Failed to run Confluence LM retrieval: {e}")


def run_jira_lm():
    """
    Jira LM Retrieval (STUB)
    TODO:
    - Gunakan jira_items_coll
    - Tentukan vector vs hybrid
    - Integrasi rerank jika perlu
    """
    print("\n[MODE] Jira LM selected\n")
    print("[INFO] Jira LM retrieval is NOT IMPLEMENTED yet.")
    print("[TODO] This pipeline will be added in the future.\n")


def main():
    print("=" * 70)
    print("ALL RETRIEVAL ENTRY POINT")
    print("=" * 70)
    print("Pilih mode retrieval:")
    print("1. MPM ORP (default)")
    print("2. MPM LM (COMING SOON)")
    print("3. Confluence LM")
    print("4. Jira LM (COMING SOON)")
    print("-" * 70)

    choice = input("Masukkan pilihan (1-4) [default=1]: ").strip()

    # Default behavior
    if choice == "" or choice == "1":
        run_mpm_orp()
    elif choice == "2":
        run_mpm_lm()
    elif choice == "3":
        run_confluence_lm()
    elif choice == "4":
        run_jira_lm()
    else:
        print(f"[ERROR] Invalid choice: '{choice}'. Please choose 1-4.")

if __name__ == "__main__":
    main()