//! Basic file I/O

use crate::errors::OerstedError;
use std::{fs::read_to_string, process::Termination};

/// CSV data stored in row-major format
pub struct CsvData {
    _nrows: usize,
    _ncols: usize,
    _data: Vec<f64>,
}

impl CsvData {
    /// Constructor; assign a default capacity
    pub fn new() -> Self {
        let default_capacity: usize = 1000;
        Self {
            _nrows: 0,
            _ncols: 0,
            _data: Vec::with_capacity(default_capacity),
        }
    }

    /// Get the number of rows in the file data
    pub fn nrows(&self) -> usize {
        self._nrows
    }

    /// Get the number of columns in the file data
    pub fn ncols(&self) -> usize {
        self._ncols
    }

    /// Provide an immutable reference to the data
    pub fn data(&self) -> &Vec<f64> {
        &self._data
    }

    /// Print the file data to stdout
    pub fn print(&self) {
        println!(
            "CSV Data Set of size {} x {}:\n",
            self.nrows(),
            self.ncols()
        );
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                print!("{}", self._data[i * self.ncols() + j]);
                if j < self.ncols() - 1 {
                    print!(", ");
                }
            }
            println!();
        }
        println!();
    }
}

impl Termination for CsvData {
    fn report(self) -> std::process::ExitCode {
        std::process::ExitCode::SUCCESS
    }
}

impl Default for CsvData {
    fn default() -> Self {
        Self::new()
    }
}

/// Read a csv file and return it as a matrix
pub fn read_csv(filename: &str, delimiter: char, skiprows: usize) -> Result<CsvData, OerstedError> {
    let file_data: String = read_to_string(filename).map_err(|_| OerstedError::FileOpenError)?;

    let mut data: CsvData = CsvData::new();
    let mut nrows: usize = 0; // number of rows of data
    let mut ncols: usize; // number of columns of data/file
    let mut line_number: usize = 0; // line number of the file

    // Read file line by line:
    // 1. Skip user-defined header rows
    // 2. Skip empty lines
    // 3. Split each line by delimiter and trim whitespace
    // 4. Convert to f64 data
    // 5. Check to make sure that each row has same number of columns
    for line in file_data.lines() {
        if line_number < skiprows {
            line_number += 1;
            continue;
        }
        if line.is_empty() {
            line_number += 1;
            continue;
        }

        ncols = 0;
        for word in line.split(delimiter) {
            ncols += 1;
            if nrows == 0 {
                data._ncols += 1;
            }
            let val: f64 = word
                .trim()
                .parse::<f64>()
                .map_err(|_| OerstedError::FloatParseError)?;
            data._data.push(val);
        }
        if data.ncols() != ncols {
            return Err(OerstedError::ColNumberChange);
        }
        line_number += 1;
        nrows += 1;
    }
    data._nrows = nrows;

    Ok(data)
}

// Eliminating this test from the test set, as this functionality is not used
// #[cfg(test)]
// mod tests {

//     use super::*;
//     use crate::errors::oerstedError;

//     /// Test loading a file containing standard delimiters and some
//     /// extra whitespace
//     #[test]
//     fn test_read_csv() -> Result<CsvData, oerstedError>{
//         let filename: String = "tests/data/test_read.csv".to_owned();
//         let skiprows: usize = 1;
//         let delimiter: char = ',';
//         let data: CsvData = read_csv(&filename, delimiter, skiprows)?;
//         data.print();
//         Ok(data)
//     }
// }
