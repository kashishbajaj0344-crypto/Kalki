//
//  ViewModel.swift
//  TaskMaster
//

import Foundation
import Combine

class ViewModel: ObservableObject {
    @Published var items: [TodoItem] = []
    @Published var isLoading = false
    
    func fetchData() {
        isLoading = true
        // Simulate network call
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            self.items = self.loadSampleData()
            self.isLoading = false
        }
    }
    
    private func loadSampleData() -> [TodoItem] {
        return [
            TodoItem(title: "Sample Item 1"),
            TodoItem(title: "Sample Item 2"),
            TodoItem(title: "Sample Item 3")
        ]
    }
}
