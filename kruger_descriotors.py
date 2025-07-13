smiles_features_conf = ["mass", "NumAtoms", "NumBonds", "NumSingleBonds", "NumDoubleBonds", "NumTripleBonds",
                            "NumAromBonds", "AromC", "Charge", "C", "O", "H", "carboxyle", "hydroxyl", "ester", "carbonyl",
                           'BertzCT', 'Ipc', 'TPSA', 'NHOHCount', 'MolMR','VSA_EState3', 'AvgIpc',
                        'VSA_EState3', 'ExactMolWt', 'HeavyAtomMolWt', 'NumHDonors', 'Chi1', 'OC-ratio']


smiles_features_broad = ["mass", "NumAtoms", "NumBonds", "NumSingleBonds", "NumDoubleBonds", "NumTripleBonds",
                           "NumAromBonds", "AromC", "Charge", "C", "O", "H", "Cl", "N", "I", "S", "F", "P", "Si", "Br", "B",
                           "hydroxyl", "carboxyle", "ester", "amine", "amide", "carbonyl", "sulfide", "nitro", "nitrile",
                           'BertzCT', 'Ipc', 'TPSA', 'NHOHCount', 'MolMR','VSA_EState3', 'AvgIpc',
                        'VSA_EState3', 'ExactMolWt', 'HeavyAtomMolWt', 'NumHDonors', 'Chi1', 'OC-ratio']


smiles_features_geckoq = ["mass", "NumAtoms", "NumBonds", "NumSingleBonds", "NumDoubleBonds", "NumTripleBonds",
                          "NumAromBonds", "AromC", "Charge", "C", "O", "N", "ketone", "hydroperoxide", "hydroxyl",
                          "nitrate", "aldehyde", "carboxylic acid", "peroxide", "carbonylperoxynitrate",
                          "carbonylperoxynitrate", "ether", "nitro", "ester", "nitroester",
                           'BertzCT', 'Ipc', 'TPSA', 'NHOHCount', 'MolMR','VSA_EState3', 'AvgIpc',
                        'VSA_EState3', 'ExactMolWt', 'HeavyAtomMolWt', 'NumHDonors', 'Chi1', 'OC-ratio']

def external_process_smiles(smiles, smiles_features):
    full_output_vector = []
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    for smil in smiles:
        mol = Chem.MolFromSmiles(smil)
        molH = Chem.AddHs(mol)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        output_vector = []

        for feature in smiles_features:
            if feature == "mass":
                mass = Descriptors.MolWt(mol) / 1000
                output_vector.append(mass)
            elif feature == "ExactMolWt":
                value = Descriptors.ExactMolWt(mol) / 1000
                output_vector.append(value)
            elif feature == 'TPSA':
                value = Descriptors.TPSA(mol) / 320
                output_vector.append(value)
            elif feature == 'HeavyAtomMolWt':
                value = Descriptors.HeavyAtomMolWt(mol) / 1000
                output_vector.append(value)
            elif feature == 'VSA_EState3':
                value = (Descriptors.VSA_EState3(mol) + 40) / 160
                output_vector.append(value)
            elif feature == 'NumHDonors':
                value = Descriptors.NumHDonors(mol) / mol.GetNumAtoms()
                output_vector.append(value)
            elif feature == 'Chi1':
                value = Descriptors.Chi1(mol) / 30
                output_vector.append(value)
            elif feature == 'BertzCT':
                value = Descriptors.BertzCT(mol) / 3000
                output_vector.append(value)
            elif feature == 'Ipc':
                value = Descriptors.Ipc(mol) / 60000000000000
                output_vector.append(value)
            elif feature == 'NHOHCount':
                value = Descriptors.NHOHCount(mol)/13
                output_vector.append(value)
            elif feature == 'MolMR':
                value = Descriptors.MolMR(mol) / 300
                output_vector.append(value)
            elif feature == 'AvgIpc':
                value = Descriptors.AvgIpc(mol) / 5
                output_vector.append(value)
            elif feature == "NumAtoms":
                numatoms = mol.GetNumAtoms() / 70
                output_vector.append(numatoms)
            elif feature == "NumBonds":
                numbonds = mol.GetNumBonds() / 70
                output_vector.append(numbonds)
            elif feature == "NumSingleBonds":
                try:
                    numsinglebonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.SINGLE) / mol.GetNumBonds()
                    output_vector.append(numsinglebonds)
                except:
                    output_vector.append(0)
            elif feature == "NumDoubleBonds":
                try:
                    numdoublebonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE) / mol.GetNumBonds()
                    output_vector.append(numdoublebonds)
                except:
                    output_vector.append(0)
            elif feature == "NumTripleBonds":
                try:
                    numtriplebonds = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.TRIPLE) / mol.GetNumBonds()
                    output_vector.append(numtriplebonds)
                except:
                    output_vector.append(0)
            elif feature == "NumAromBonds":
                try:
                    numarombonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic()) / mol.GetNumBonds()
                    output_vector.append(numarombonds)
                except:
                    output_vector.append(0)
            elif feature == "AromC":
                aromC = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C" and atom.GetIsAromatic()) / mol.GetNumAtoms()
                output_vector.append(aromC)
            elif feature == "Charge":
                charge = (Chem.GetFormalCharge(mol) + 2) / 4
                output_vector.append(charge)
            elif feature == "OC-ratio":
                C_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
                O_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "O")
                try:
                    output_vector.append((O_count/C_count)/8)
                except:
                    output_vector.append(1)
            elif feature == "H":
                atom_count = sum(1 for atom in molH.GetAtoms() if atom.GetSymbol() == "H") / molH.GetNumAtoms()
                output_vector.append(atom_count)
            elif len(feature) < 3:
                # if small, assume it's an element symbol and count the number of atoms
                atom_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == feature) / mol.GetNumAtoms()
                output_vector.append(atom_count)
            elif feature == "carboxyle":
                pattern = Chem.MolFromSmarts("C(=O)O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "ester":
                pattern = Chem.MolFromSmarts("C(=O)OC")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "hydroxyl":
                pattern = Chem.MolFromSmarts("[OH]")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "carbonyl":
                pattern = Chem.MolFromSmarts("C=O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "amine":
                pattern = Chem.MolFromSmarts("N")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "amide":
                pattern = Chem.MolFromSmarts("C(=O)N")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "sulfide":
                pattern = Chem.MolFromSmarts("S")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "nitro":
                pattern = Chem.MolFromSmarts("N(=O)=O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "nitrile":
                pattern = Chem.MolFromSmarts("C#N")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "ketone":
                pattern = Chem.MolFromSmarts("C(=O)C")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "hydroperoxide":
                pattern = Chem.MolFromSmarts("OO")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "nitrate":
                pattern = Chem.MolFromSmarts("O[N+](=O)[O-]")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "aldehyde":
                pattern = Chem.MolFromSmarts("C=O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "carboxylic acid":
                pattern = Chem.MolFromSmarts("C(=O)O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "peroxide":
                pattern = Chem.MolFromSmarts("O-O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "carbonylperoxynitrate":
                pattern = Chem.MolFromSmarts("C(=O)OON(=O)=O")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "ether":
                pattern = Chem.MolFromSmarts("C-O-C")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            elif feature == "nitroester":
                pattern = Chem.MolFromSmarts("C(=O)ON(=O)")
                matches = mol.GetSubstructMatches(pattern)
                output_vector.append(len(matches) / 10)
            else:
                raise ValueError("Feature " + feature + " not found.")

        full_output_vector.append(output_vector)

    return full_output_vector