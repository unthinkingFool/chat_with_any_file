#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

//implementing binary search algorithm

int binary_search(vector<int>& a, int target) {
    int low = 0;
    int high = a.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (a[mid] == target) {
            return mid;
        }
        if (a[mid] < target) {
            low = mid + 1;
        }
        else {
            high = mid - 1;
        }
    }
    return -1;
}   

int main() {
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }
    sort(a.begin(), a.end());
    int target;
    cin >> target;
    int index = binary_search(a, target);
    if (index != -1) {
        cout << "Element found at index " << index << endl;
    }
    else {
        cout << "Element not found" << endl;
    }

    // the sorted array
    for (int i = 0; i < n; i++) {
        cout << a[i] << " ";
    }
    cout << endl;
    return 0;
}