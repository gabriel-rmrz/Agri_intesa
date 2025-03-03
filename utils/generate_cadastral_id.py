def generate_cadastral_id(cod_comune, foglio, particella):
  # Ensure foglio is formatted with leading zeroes to be at least 4 characters.
  foglio_str = str(foglio).zfill(4)

  cadastral_id = f"CadastralParcel.IT.AGE.PLA.{cod_comune}_{foglio_str}00.{particella}"
  return cadastral_id

