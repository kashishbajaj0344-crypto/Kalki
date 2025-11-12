//
//  Monetization.swift
//  In-App Purchases
//

import StoreKit

class IAPManager: NSObject, ObservableObject {
    @Published var products: [SKProduct] = []
    
    func fetchProducts() {
        let productIDs: Set<String> = ["com.yourapp.premium", "com.yourapp.coins"]
        let request = SKProductsRequest(productIdentifiers: productIDs)
        request.delegate = self
        request.start()
    }
    
    func purchase(_ product: SKProduct) {
        let payment = SKPayment(product: product)
        SKPaymentQueue.default().add(payment)
    }
}

extension IAPManager: SKProductsRequestDelegate {
    func productsRequest(_ request: SKProductsRequest, didReceive response: SKProductsResponse) {
        DispatchQueue.main.async {
            self.products = response.products
        }
    }
}
