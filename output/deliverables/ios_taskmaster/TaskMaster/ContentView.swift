//
//  ContentView.swift
//  TaskMaster
//

import SwiftUI

struct ContentView: View {
    @State private var items: [TodoItem] = []
    @State private var newItemTitle = ""
    
    var body: some View {
        NavigationView {
            VStack {
                // Add item section
                HStack {
                    TextField("New item", text: $newItemTitle)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                    
                    Button(action: addItem) {
                        Image(systemName: "plus.circle.fill")
                            .font(.title)
                    }
                    .disabled(newItemTitle.isEmpty)
                }
                .padding()
                
                // Items list
                List {
                    ForEach(items) { item in
                        HStack {
                            Image(systemName: item.isCompleted ? "checkmark.circle.fill" : "circle")
                                .foregroundColor(item.isCompleted ? .green : .gray)
                            Text(item.title)
                            Spacer()
                        }
                        .contentShape(Rectangle())
                        .onTapGesture {
                            toggleItem(item)
                        }
                    }
                    .onDelete(perform: deleteItems)
                }
            }
            .navigationTitle("TaskMaster")
        }
    }
    
    private func addItem() {
        let newItem = TodoItem(title: newItemTitle)
        items.append(newItem)
        newItemTitle = ""
    }
    
    private func toggleItem(_ item: TodoItem) {
        if let index = items.firstIndex(where: { $0.id == item.id }) {
            items[index].isCompleted.toggle()
        }
    }
    
    private func deleteItems(at offsets: IndexSet) {
        items.remove(atOffsets: offsets)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
