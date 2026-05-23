"""
Evaluation tests using golden dataset
Validates subsystem outputs against expected results
"""

import os
import sys
import json
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class TestGoldenDataset(unittest.TestCase):
    """Validate the golden dataset structure and content"""
    
    def setUp(self):
        dataset_path = os.path.join(os.path.dirname(__file__), '../../data/golden_dataset.json')
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
    
    def test_dataset_not_empty(self):
        """Test that dataset contains test cases"""
        self.assertGreater(len(self.dataset), 0)
    
    def test_all_cases_have_required_fields(self):
        """Test that each case has the required structure"""
        for case in self.dataset:
            self.assertIn('id', case, f"Case missing 'id': {case}")
            self.assertIn('subsystem', case, f"Case {case['id']} missing 'subsystem'")
            self.assertIn('input', case, f"Case {case['id']} missing 'input'")
            self.assertIn('expected_output', case, f"Case {case['id']} missing 'expected_output'")
    
    def test_valid_subsystem_values(self):
        """Test that subsystem values are valid"""
        valid_subsystems = {'clinical_diagnosis', 'prescription_verification', 
                           'medical_records', 'appointment', 'research'}
        for case in self.dataset:
            self.assertIn(
                case['subsystem'], valid_subsystems,
                f"Case {case['id']} has invalid subsystem: {case['subsystem']}"
            )
    
    def test_all_subsystems_represented(self):
        """Test that all 5 subsystems have test cases"""
        subsystems = set(case['subsystem'] for case in self.dataset)
        expected = {'clinical_diagnosis', 'prescription_verification', 
                   'medical_records', 'appointment', 'research'}
        self.assertEqual(subsystems, expected, 
                        f"Missing test cases for: {expected - subsystems}")
    
    def test_clinical_cases_have_expected_diagnosis(self):
        """Test that clinical diagnosis cases have expected output structure"""
        for case in self.dataset:
            if case['subsystem'] == 'clinical_diagnosis':
                expected = case['expected_output']
                self.assertIn('primary_diagnosis', expected, 
                            f"Case {case['id']} missing primary_diagnosis")
                self.assertIn('urgency', expected,
                            f"Case {case['id']} missing urgency")
                self.assertIn('key_tests', expected,
                            f"Case {case['id']} missing key_tests")
    
    def test_prescription_cases_have_decision(self):
        """Test that prescription cases have decision field"""
        for case in self.dataset:
            if case['subsystem'] == 'prescription_verification':
                self.assertIn('decision', case['expected_output'],
                            f"Case {case['id']} missing decision")
    
    def test_dataset_json_valid(self):
        """Test that dataset is valid JSON"""
        # Already loaded successfully in setUp
        pass


if __name__ == '__main__':
    unittest.main()
